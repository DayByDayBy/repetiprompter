"""Unified runner for chain and tree generation modes."""

import signal
import sys
from datetime import datetime
from typing import Optional

from .models import RunConfig, NodeOutput, TopologyMode
from .output_writer import JSONLWriter
from .ollama_interface import OllamaClient
from .framing_strategies import get_strategy, FramingStrategy
from .reminder import ReminderRegime
from .temperature_regime import create_temperature_regime, TemperatureRegime


class Runner:
    """
    Unified runner that orchestrates generation.
    
    Supports both chain (linear) and tree (branching) topologies.
    """
    
    def __init__(self, config: RunConfig):
        self.config = config
        self.run_id = config.run_identity.run_id
        
        self.client = OllamaClient(
            model_name=config.model.model_name,
            max_retries=3
        )
        
        self.framing = get_strategy(
            config.prompting.framing_strategy,
            config.prompting.custom_prefix,
            config.prompting.custom_suffix
        )
        
        self.reminder = ReminderRegime(
            config.reminder,
            config.prompting.initial_prompt
        )
        
        self.temp_regime = create_temperature_regime(config.temperature_regime)
        
        self._interrupted = False
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self):
        """Setup graceful shutdown on Ctrl+C."""
        def handler(sig, frame):
            print("\n⚠ Interrupted - flushing output and exiting...")
            self._interrupted = True
        signal.signal(signal.SIGINT, handler)
    
    def run(self) -> str:
        """
        Run generation based on topology mode.
        
        Returns:
            Path to output JSONL file
        """
        if self.config.topology.mode == TopologyMode.CHAIN:
            return self.run_chain()
        else:
            return self.run_tree()
    
    def run_chain(self) -> str:
        """Run linear chain generation."""
        steps = self.config.topology.chain.steps
        
        print(f"Starting chain run: {steps} steps")
        print(f"  Model: {self.config.model.model_name}")
        print(f"  Output: {self.config.output.output_dir}/{self.run_id}.jsonl")
        
        with JSONLWriter(self.config) as writer:
            current_prompt = self.config.prompting.initial_prompt
            parent_id = None
            
            for step in range(steps):
                if self._interrupted:
                    break
                
                node = self._generate_node(
                    writer=writer,
                    prompt=current_prompt,
                    parent_id=parent_id,
                    depth=step,
                    sibling_index=0
                )
                
                if node is None:
                    print(f"  ✗ Step {step} failed, stopping")
                    break
                
                print(f"  ✓ Step {step}/{steps-1}: {node.eval_count or '?'} tokens")
                
                current_prompt = node.response
                parent_id = node.node_id
            
            output_path = str(writer.output_path)
        
        print(f"Chain complete: {writer.nodes_written} nodes written")
        return output_path
    
    def run_tree(self) -> str:
        """Run recursive tree generation."""
        depth = self.config.topology.tree.depth
        branching = self.config.topology.tree.branching_factor
        branch_all = self.config.topology.tree.branch_all_nodes
        
        print(f"Starting tree run: depth={depth}, branching={branching}")
        print(f"  Model: {self.config.model.model_name}")
        print(f"  Output: {self.config.output.output_dir}/{self.run_id}.jsonl")
        
        with JSONLWriter(self.config) as writer:
            self._generate_tree_recursive(
                writer=writer,
                prompt=self.config.prompting.initial_prompt,
                parent_id=None,
                current_depth=0,
                max_depth=depth,
                branching=branching,
                branch_all=branch_all,
                sibling_index=0
            )
            output_path = str(writer.output_path)
        
        print(f"Tree complete: {writer.nodes_written} nodes written")
        return output_path
    
    def _generate_tree_recursive(
        self,
        writer: JSONLWriter,
        prompt: str,
        parent_id: Optional[str],
        current_depth: int,
        max_depth: int,
        branching: int,
        branch_all: bool,
        sibling_index: int
    ) -> None:
        """Recursively generate tree nodes."""
        if self._interrupted:
            return
        
        if current_depth >= max_depth:
            return
        
        node = self._generate_node(
            writer=writer,
            prompt=prompt,
            parent_id=parent_id,
            depth=current_depth,
            sibling_index=sibling_index
        )
        
        if node is None:
            return
        
        print(f"  {'  ' * current_depth}✓ d={current_depth} s={sibling_index}: {node.eval_count or '?'} tokens")
        
        num_children = branching if (branch_all or current_depth == 0) else 1
        
        for i in range(num_children):
            if self._interrupted:
                break
            self._generate_tree_recursive(
                writer=writer,
                prompt=node.response,
                parent_id=node.node_id,
                current_depth=current_depth + 1,
                max_depth=max_depth,
                branching=branching,
                branch_all=branch_all,
                sibling_index=i
            )
    
    def _generate_node(
        self,
        writer: JSONLWriter,
        prompt: str,
        parent_id: Optional[str],
        depth: int,
        sibling_index: int
    ) -> Optional[NodeOutput]:
        """Generate a single node and write it to output."""
        reminder_result = self.reminder.check()
        
        if reminder_result.fired:
            prompt = self.reminder.apply_to_prompt(prompt, reminder_result)
        
        framed_prompt, prefix, suffix = self.framing.frame(prompt)
        
        temperature = self.temp_regime.get_temperature(depth)
        
        try:
            result = self.client.generate(
                prompt=framed_prompt,
                temperature=temperature,
                top_p=self.config.model.top_p,
                repeat_penalty=self.config.model.repeat_penalty,
                seed=self.config.model.seed
            )
        except Exception as e:
            print(f"  ✗ Generation error: {e}")
            return None
        
        node_id = writer.generate_node_id(parent_id, sibling_index)
        step_index = writer.next_step_index()
        
        node = NodeOutput(
            run_id=self.run_id,
            node_id=node_id,
            parent_id=parent_id,
            depth=depth,
            step_index=step_index,
            sibling_index=sibling_index,
            prompt=prompt,
            prefix=prefix,
            suffix=suffix,
            reminder_fired=reminder_result.fired,
            reminder_content=reminder_result.content if reminder_result.fired else None,
            response=result.response,
            prompt_eval_count=result.prompt_eval_count,
            eval_count=result.eval_count,
            prompt_eval_duration_ms=result.prompt_eval_duration_ms,
            eval_duration_ms=result.eval_duration_ms,
            temperature=temperature,
            model_name=self.config.model.model_name,
            framing_strategy=self.config.prompting.framing_strategy.value
        )
        
        writer.write_node(node)
        return node


def run_from_config(config: RunConfig) -> str:
    """Convenience function to run from a config object."""
    runner = Runner(config)
    return runner.run()