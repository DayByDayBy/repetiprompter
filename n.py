""" just a way to quickly check number of generations, so i 
    can punch in generation times and get a rough idea of how 
    long to expect a given model to take to run the script
  
    numbers are most likely from the last time i checked a model
"""

print('\n')

 
chain_length = 4
recursion_depth = 4
average_generation_time_in_seconds = 15

total_chains = sum(chain_length ** i for i in range(recursion_depth))
total_prompt_response_actions = total_chains * (chain_length-1)

print(f'chains: {total_chains} \nresponses: {total_prompt_response_actions}')
print('\n')


hours = (total_prompt_response_actions * average_generation_time_in_seconds)/3600
minutes = (total_prompt_response_actions * average_generation_time_in_seconds)/60

days = hours/24
weeks =days/7


print(f'minutes: {minutes} \nhours: {hours} \ndays: {days} \nweeks: {weeks}')



"""
fastest while-reliable model i've used so far was 1-2s (usually closer to 1 but with some markedly longer cycles)
that one was about 18hrs to run 6 by 6 - model was "stablelm2:zephyr"
i also ran tinyllama, which was similarly quick per iteration, but that model isn't really 
suited to the task, and unsurprisingly often goes a bit nuts
"""






# a = 51145/425
# b = a/121.73447603943724

# print(f'b: {b}')

# 420.1357057094574


