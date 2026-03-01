# PVS Parameters
## CASE1 Parameters
- n_values = 4
- n_actions = 2
- n_agents = 30 
- agent_groups = {"ex_low": agent_ids[:20], "ex_high": agent_ids[20:]}
- PriPs are equal to agent_groups. So group "ex_low" have low PriP (i.e. low egalitarianism pref)
- hcva2 "Consensus p is: 1.55"

## CASE2 Parameters
- n_values = 4
- n_actions = 2
- n_agents = 30 
- agent_groups = {"ex_low": agent_ids[:16], "ex_high" agent_ids[16:]}

# Principle Parameters
## PriP1 Parameters
- agent_groups = {"ex_low": agent_ids[:20], "ex_high": agent_ids[20:]}
## Prip2 Parameters
- agent_groups = {"ex_low": agent_ids[:16], "ex_high": agent_ids[16:]}

# Results
You can find the results for each case in the corresponding folder (`results/placeholder_results/CASE[X]`)
## CASE1 x PriP1 Results
- KEY = {0: Util, 1: "HCVA2", 2: "T", 3: "Egal"}
- Worst Welfare: 0.373, 0.280, 0.282, 0.290 (3 d.p)
- Gini Coeff: 0.396, 0.383, 0.384, 0.405 (3 d.p)
- Number of agents envious: 14, 16, 16, 0

## CASE2 x PriP1 Results
- KEY = {0: