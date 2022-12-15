from numpy import exp, sqrt, log
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

s0 = 100
T = 1
sigma = 2
gamma = 0.005
k = 1.5
A = 140
num_steps = 200
dt = T / num_steps
num_paths = 10000

# results of simulations
inv_profit, inv_final_q = [], []
sym_profit, sym_final_q = [], []
for _ in range(num_paths):
    """
    parameters along the trajectory
    prefix inv: inventory strategy
    prefix sym: symmetric strategy
    """
    s, r, ra, rb, inv_x, inv_q, inv_w, sym_x, sym_q, sym_w = \
        [np.empty(num_steps + 1) for _ in range(10)]
    s[0], r[0], ra[0], rb[0] = s0, s0, s0, s0
    inv_x[0], inv_q[0], inv_w[0] = 0, 0, 0
    sym_x[0], sym_q[0], sym_w[0] = 0, 0, 0

    for i in range(1, num_steps + 1):
        # update state variables
        z = np.random.standard_normal()
        s[i] = s[i - 1] + sigma * z * sqrt(dt)
        r[i] = s[i] - inv_q[i - 1] * gamma * sigma ** 2 * (T - i * dt)
        spread = gamma * sigma ** 2 * (T - i * dt) + 2 / gamma * log(1 + gamma / k)
        rb[i] = r[i] - spread / 2
        ra[i] = r[i] + spread / 2

        # compute bid/ask prices
        inv_delta_a = ra[i] - s[i]
        inv_delta_b = s[i] - rb[i]
        sym_delta = spread / 2

        # compute trading intensity
        inv_lambda_a = A * exp(-k * inv_delta_a)
        inv_lambda_b = A * exp(-k * inv_delta_b)
        sym_lambda = A * exp(-k * sym_delta)

        # generate random vars
        inv_prob_a = inv_lambda_a * dt
        inv_prob_b = inv_lambda_b * dt
        sym_prob = sym_lambda * dt

        # decide inventory change based on random variable
        ask_rand = np.random.random()
        buy_rand = np.random.random()
        inv_dNa = 1 if ask_rand < inv_prob_a else 0
        inv_dNb = 1 if buy_rand < inv_prob_b else 0
        sym_dNa = 1 if ask_rand < sym_prob else 0
        sym_dNb = 1 if buy_rand < sym_prob else 0

        # update agents' wealth
        inv_q[i] = inv_q[i - 1] - inv_dNa + inv_dNb
        inv_x[i] = inv_x[i - 1] + ra[i] * inv_dNa - rb[i] * inv_dNb
        inv_w[i] = inv_x[i] + inv_q[i] * s[i]
        sym_q[i] = sym_q[i - 1] - sym_dNa + sym_dNb
        sym_x[i] = sym_x[i - 1] + (s[i] + spread / 2) * sym_dNa - (s[i] - spread / 2) * sym_dNb
        sym_w[i] = sym_x[i] + sym_q[i] * s[i]
    inv_profit.append(inv_w[-1])
    inv_final_q.append(inv_q[-1])
    sym_profit.append(sym_w[-1])
    sym_final_q.append(sym_q[-1])

# plots
colors = sns.color_palette("Set2", 8)
plt.figure(figsize=(8, 4))
plt.title('market making process')
sns.set_style("darkgrid")
plt.plot(s, color=colors[0], label='mid price')
plt.plot(r, color=colors[1], label='indifference price')
plt.plot(ra, '.', color=colors[3], label='price asked')
plt.plot(rb, '.', color=colors[4], label='price bid')
plt.legend()
plt.savefig(f"./docs/pics/gamma{gamma}_process.png")

plt.figure()
bins = np.linspace(0, 150, 100)
plt.hist(inv_profit, bins, alpha=0.5, color=colors[0], label='inventory strategy profits')
plt.hist(sym_profit, bins, alpha=0.5, color=colors[1], label='symmetric strategy profits')
plt.legend(loc='upper right')
plt.savefig(f"./docs/pics/gamma{gamma}_profitHist.png")

plt.figure()
plt.title(f'sample path of inventory changes')
sns.set_style("darkgrid")
plt.plot(inv_q, color=colors[0], label=f'inventory change of inventory strategy, gamma={gamma}')
plt.plot(sym_q, color=colors[1], label=f'inventory change of symmetric strategy, gamma={gamma}')
plt.legend()
plt.savefig(f"./docs/pics/gamma{gamma}_inventory.png")
#
# plt.figure()
# plt.title(f'histogram of symmetric strategy profits in {num_paths} simulations')
# sns.histplot(sym_profit, bins=30)

plt.figure()
plt.figure('pnl of one simulation')
plt.plot(inv_w, color=colors[0], label='inventory strategy pnl')
plt.plot(sym_w, color=colors[1], label='symmetric strategy pnl')
plt.legend()
plt.savefig(f"./docs/pics/gamma{gamma}_pnl.png")
plt.show()

print("Inventory strategy report")
print(f'spread: {spread: .3f}')
print(f'mean of profit: {np.mean(inv_profit): .3f}')
print(f'std of profit: {np.std(inv_profit): .3f}')
print(f'mean of final inventory: {np.mean(inv_final_q): .3f}')
print(f'std of final inventory: {np.std(inv_final_q): .3f}')

print("\nSymmetric Strategy report")
print(f'spread: {spread: .3f}')
print(f'mean of profit: {np.mean(sym_profit): .3f}')
print(f'std of profit: {np.std(sym_profit): .3f}')
print(f'mean of final inventory: {np.mean(sym_final_q): .3f}')
print(f'std of final inventory: {np.std(sym_final_q): .3f}')
