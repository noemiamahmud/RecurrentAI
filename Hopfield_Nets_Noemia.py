import random

N = 16
Q = 0.0

RUNS = 5
P_VALUES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
MAX_ITERS = 5000


def parse_pattern(bits: str):
    bits = bits.strip()
    if len(bits) != N:
        raise ValueError("pattern must be length 16")
    return [int(c) for c in bits]


def add_noise(pattern, p_noise):
    out = pattern[:]
    for i in range(len(out)):
        if random.random() < p_noise:
            out[i] = 1 - out[i]
    return out


def hamming(a, b):
    d = 0
    for i in range(len(a)):
        if a[i] != b[i]:
            d += 1
    return d


def fmt_pattern(p):
    s = "".join(str(x) for x in p)
    return s[:8] + " " + s[8:]


def energy_went_up(energies):
    for i in range(1, len(energies)):
        if energies[i] > energies[i - 1] + 1e-12:
            return True
    return False


class Hoppynets:
    def __init__(self, n=N, q=Q):
        self.n = n
        self.q = q
        self.state = [0] * n

        self.w = []
        for _ in range(n):
            self.w.append([0.0] * n)

    def erase_memory(self):
        for i in range(self.n):
            for j in range(self.n):
                self.w[i][j] = 0.0

    def set_state(self, pattern01):
        self.state = pattern01[
            :]

    def train_one(self, pattern01):
        # same rule you had: if bits match -> +1 else -1 (only i<j, mirror it)
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if pattern01[i] == pattern01[j]:
                    delta = 1.0
                else:
                    delta = -1.0
                self.w[i][j] += delta
                self.w[j][i] += delta

    def train_all(self, patterns01):
        for p in patterns01:
            self.train_one(p)

    def net_input(self, idx, state=None):
        a = self.state if state is None else state
        total = 0.0
        row = self.w[idx]
        for j in range(self.n):
            if j != idx:
                total += row[j] * a[j]
        return total

    def next_activation(self, idx, state=None):
        return 1 if self.net_input(idx, state) > self.q else 0

    def desired_activations(self):
        old = self.state[:]
        desired = [0] * self.n
        for i in range(self.n):
            desired[i] = self.next_activation(i, old)
        return desired

    def hopfield_async_step(self):
        i = random.randrange(self.n)
        self.state[i] = self.next_activation(i, self.state)
        desired_now = self.desired_activations()
        return self.state == desired_now

    def synchronous_step(self):
        old = self.state[:]
        new_state = [0] * self.n
        for i in range(self.n):
            new_state[i] = self.next_activation(i, old)
        self.state = new_state

    def energy(self, state=None):
        a = self.state if state is None else state
        total = 0.0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                total += self.w[i][j] * a[i] * a[j]
        return -total


def run_async_until_settle(net: Hoppynets, max_iters=MAX_ITERS):
    energies = []
    for t in range(max_iters):
        energies.append(net.energy())
        did_settle = net.hopfield_async_step()
        if did_settle:
            energies.append(net.energy())
            return True, t + 1, energies
    return False, max_iters, energies


def run_sync_until_settle(net: Hoppynets, max_iters=MAX_ITERS):
    energies = []
    seen = set()

    for t in range(max_iters):
        s = tuple(net.state)
        if s in seen:
            return False, t, energies
        seen.add(s)

        energies.append(net.energy())
        old = net.state[:]
        net.synchronous_step()

        if net.state == old:
            energies.append(net.energy())
            return True, t + 1, energies

    return False, max_iters, energies


def run_async_k_until_settle(net: Hoppynets, k=4, max_iters=MAX_ITERS):
    energies = []
    for t in range(max_iters):
        energies.append(net.energy())

        desired = net.desired_activations()
        idxs = random.sample(range(net.n), k=min(k, net.n))
        for i in idxs:
            net.state[i] = desired[i]

        if net.state == desired:
            energies.append(net.energy())
            return True, t + 1, energies

    return False, max_iters, energies


def print_report(title, results_by_p, p_values):
    print("\n" + title)
    for p in p_values:
        total_runs = results_by_p[p]["total"]
        settled_runs = results_by_p[p]["settled"]
        failures = results_by_p[p]["fail"]
        up_runs = results_by_p[p]["energy_up"]

        if settled_runs > 0:
            mean_hd = results_by_p[p]["hd_sum"] / settled_runs
            mean_it = results_by_p[p]["it_sum"] / settled_runs
        else:
            mean_hd = "N/A"
            mean_it = "N/A"

        print(f"\np = {p}")
        print("  mean hamming distance:", mean_hd)
        print("  mean iterations to settle:", mean_it)
        print("  failures to settle:", f"{failures}/{total_runs}")
        print("  runs where energy increased:", f"{up_runs}/{total_runs}")


def run_condition_block(net, patterns, p_values, runs_per_condition, mode):
    # mode: "async" | "sync" | "async_k"
    results = {}
    for p in p_values:
        results[p] = {"hd_sum": 0, "it_sum": 0, "settled": 0, "fail": 0, "energy_up": 0, "total": 0}

    for clean in patterns:
        for p_noise in p_values:
            for _ in range(runs_per_condition):
                results[p_noise]["total"] += 1

                test_pat = add_noise(clean, p_noise)
                net.set_state(test_pat)

                if mode == "async":
                    did_settle, iters, energies = run_async_until_settle(net, MAX_ITERS)
                elif mode == "sync":
                    did_settle, iters, energies = run_sync_until_settle(net, MAX_ITERS)
                elif mode == "async_k":
                    did_settle, iters, energies = run_async_k_until_settle(net, k=4, max_iters=MAX_ITERS)
                else:
                    raise ValueError("unknown mode")

                if energy_went_up(energies):
                    results[p_noise]["energy_up"] += 1

                if did_settle:
                    results[p_noise]["settled"] += 1
                    results[p_noise]["hd_sum"] += hamming(net.state, clean)
                    results[p_noise]["it_sum"] += iters

                    # print("DEBUG final:", fmt_pattern(net.state), "iters=", iters)
                else:
                    results[p_noise]["fail"] += 1

    return results


def part2(net: Hoppynets):
    def rand_pat():
        p = []
        for _ in range(N):
            p.append(1 if random.random() < 0.5 else 0)
        return p

    pats3 = [rand_pat() for _ in range(3)]
    net.erase_memory()
    net.train_all(pats3)

    res3 = run_condition_block(net, pats3, P_VALUES, RUNS, mode="async")
    print_report("Part 2: 3 random training patterns (async)", res3, P_VALUES)

    p_only = [0.2]

    pat4 = rand_pat()
    pats4 = pats3[:] + [pat4]
    net.train_one(pat4)

    res4 = run_condition_block(net, pats4, p_only, RUNS, mode="async")
    print_report("Part 2: 4 random patterns (p=0.2 only, async)", res4, p_only)

    pat5 = rand_pat()
    pats5 = pats4[:] + [pat5]
    net.train_one(pat5)

    res5 = run_condition_block(net, pats5, p_only, RUNS, mode="async")
    print_report("Part 2: 5 random patterns (p=0.2 only, async)", res5, p_only)


def part3(net: Hoppynets):
    base = [1] * 8 + [0] * 8
    train6 = []
    for _ in range(6):
        train6.append(add_noise(base, 0.125))

    net.erase_memory()
    net.train_all(train6)

    res = run_condition_block(net, train6, [0.0], runs_per_condition=1, mode="async")
    print_report("Part 3: related patterns (p=0.0 only, async)", res, [0.0])

    print("\nPart 3B: Final settled states for each training pattern:")
    for idx, p in enumerate(train6):
        net.set_state(p)
        did_settle, iters, _ = run_async_until_settle(net, MAX_ITERS)
        print(f"  S{idx+1} start={fmt_pattern(p)}  final={fmt_pattern(net.state)}  settled={did_settle}  iters={iters}")


def part4(net: Hoppynets, walsh):
    net.erase_memory()
    net.train_all(walsh)

    p_only = [0.2]

    res_async = run_condition_block(net, walsh, p_only, RUNS, mode="async")
    print_report("Part 4: Hopfield async (p=0.2)", res_async, p_only)

    res_sync = run_condition_block(net, walsh, p_only, RUNS, mode="sync")
    print_report("Part 4: synchronous (p=0.2)", res_sync, p_only)

    res_k = run_condition_block(net, walsh, p_only, RUNS, mode="async_k")
    print_report("Part 4: async variant (update 4 nodes/step, p=0.2)", res_k, p_only)


if __name__ == "__main__":
    random.seed(0)

    walsh = [
        parse_pattern("1111111100000000"),
        parse_pattern("1111000011110000"),
        parse_pattern("1100110011001100"),
        parse_pattern("1010101010101010"),
    ]

    net = Hoppynets()

    net.erase_memory()
    net.train_all(walsh)

    res1 = run_condition_block(net, walsh, P_VALUES, RUNS, mode="async")
    print_report("Part 1: Walsh patterns (async)", res1, P_VALUES)

    part2(net)
    part3(net)
    part4(net, walsh)


"""
Write-Up

Part 1

1A: running 120 simulations (4 trained patterns X 6 probability conditions X 5 runs per pattern)

1B: Summarize these data over patterns within each condition (e.g., report means of hamming distance
and time-to-settle in each condition). Report the number of times in each condition the network failed
to settle (if ever). 

For each Walsh training pattern, I made test patterns by flipping each bit independently with probability
p = 0.0, 0.1, 0.2, 0.3, 0.4, 0.5. For each (pattern, p) condition I ran 5 trials (4 patterns × 6 p-values × 5 runs = 120 total).
The network was run until it settled, or until it hit the max iterations 

Results:
p = 0.0
  mean hamming distance: 0.0
  mean iterations to settle: 1.0
  failures to settle: 0/20
  runs where energy increased: 0/20

p = 0.1
  mean hamming distance: 1.2
  mean iterations to settle: 32.95
  failures to settle: 0/20
  runs where energy increased: 0/20

p = 0.2
  mean hamming distance: 2.65
  mean iterations to settle: 32.6
  failures to settle: 0/20
  runs where energy increased: 0/20

p = 0.3
  mean hamming distance: 4.3
  mean iterations to settle: 36.55
  failures to settle: 0/20
  runs where energy increased: 0/20

p = 0.4
  mean hamming distance: 6.95
  mean iterations to settle: 41.15
  failures to settle: 0/20
  runs where energy increased: 0/20

p = 0.5
  mean hamming distance: 7.2
  mean iterations to settle: 30.45
  failures to settle: 0/20
  runs where energy increased: 0/20

As noise increases, the final state is usually farther from the original training pattern,
since Hamming distance increases. The network never failed to settle in these conditions either. 
Time did settle did show some increase 

Does energy always decline (or stay the same) on successive iterations, or does it ever go up from
one iteration to the next?

For Part 1, energy never increased from one iteration to the next across any run/ So these tests, energy always
declined or stayed the same on successive iterations.


Part 2

Training on random patterns: 
Why do the results differ from those you obtained in Part 1? 
I tested the network the same way as Part 1B, 

Part 2: 3 random training patterns (async)

p = 0.0
  mean hamming distance: 0.6666666666666666
  mean iterations to settle: 8.8
  failures to settle: 0/15
  runs where energy increased: 0/15

p = 0.1
  mean hamming distance: 2.2666666666666666
  mean iterations to settle: 36.13333333333333
  failures to settle: 0/15
  runs where energy increased: 0/15

p = 0.2
  mean hamming distance: 3.2
  mean iterations to settle: 23.6
  failures to settle: 0/15
  runs where energy increased: 0/15

p = 0.3
  mean hamming distance: 5.4
  mean iterations to settle: 32.06666666666667
  failures to settle: 0/15
  runs where energy increased: 0/15

p = 0.4
  mean hamming distance: 10.733333333333333
  mean iterations to settle: 33.13333333333333
  failures to settle: 0/15
  runs where energy increased: 0/15

p = 0.5
  mean hamming distance: 8.466666666666667
  mean iterations to settle: 40.46666666666667
  failures to settle: 0/15
  runs where energy increased: 0/15
  
Overall, the performance had worsened for random patterns as opposed to the given walsh patterns. 
the network sometimes did not return exactly to the original random training pattern even when
having started from it. adding noise also caused the hamming distance to jump up very quickly. The 
waslh patterns also did not overlap as much as the random patterns, so the weight updates tend to interfere 
with each other.


On these latter tests it is only necessary to run the 0.2 probability-of-change condition. What happens to performance 
as you train the network on more patterns?

Results of adding pattern 4 and 5 (p = 0.2 only)
Part 2: 4 random patterns (p=0.2 only, async)

p = 0.2
  mean hamming distance: 3.9
  mean iterations to settle: 39.5
  failures to settle: 0/20
  runs where energy increased: 0/20

Part 2: 5 random patterns (p=0.2 only, async)

p = 0.2
  mean hamming distance: 3.28
  mean iterations to settle: 35.2
  failures to settle: 0/25
  runs where energy increased: 0/25


Adding more random patterns increases interference and makes recall less reliable.
Going from 3 to 4 patterns at p = 0.2 increased the mean Hamming distance. With 5 patterns,
it dropped slightly, but that’s likely just randomness in the specific patterns. Overall, 
adding more random patterns makes recall less reliable compared to the Walsh set.


Part 3

B) Test your network with the original Training patterns (i.e., the probability-of-change = 0 condition 
from Parts 1 and 2 only). What do you notice about the state(s) into which the network usually settles? 
Why does it do this? To what might this phenomenon correspond psychologically?

Result:Part 3: related patterns (p=0.0 only, async)

p = 0.0
  mean hamming distance: 1.0
  mean iterations to settle: 18.666666666666668
  failures to settle: 0/6
  runs where energy increased: 0/6

Part 3B: Final settled states for each training pattern (start -> final):
  S1 start=11111111 00000000  final=11111111 00000000  settled=True  iters=1
  S2 start=01111111 00001000  final=11111111 00000000  settled=True  iters=17
  S3 start=11111111 00000001  final=11111111 00000000  settled=True  iters=2
  S4 start=10111111 00000000  final=11111111 00000000  settled=True  iters=5
  S5 start=11111110 00000000  final=11111111 00000000  settled=True  iters=10
  S6 start=11111011 00000000  final=11111111 00000000  settled=True  iters=10


Even though the network was trained on the six variants, (not the base), it usually settled 
into the same single pattern: 1111111100000000, which was the base pattern. This happens because
the six training patterns were all somewhat similar to the base, so this reinforced their shared 
features. 

Psychologically, this seems to represent categorical memory. A person may see many similar examples
of a pattern, and then create a general gist-like representation stored in memory. 


Part 4
1B) experiment with both synchronous updating (updating all the nodes in parallel) and with various other
approaches to asynchronous updating (e.g., randomly choosing n nodes to update on each iteration).

results:Part 4: Hopfield async (p=0.2)

p = 0.2
  mean hamming distance: 3.3
  mean iterations to settle: 28.2
  failures to settle: 0/20
  runs where energy increased: 0/20

Part 4: synchronous (p=0.2)

p = 0.2
  mean hamming distance: 2.0
  mean iterations to settle: 2.3333333333333335
  failures to settle: 14/20
  runs where energy increased: 1/20

Part 4: async variant (update 4 nodes/step, p=0.2)

p = 0.2
  mean hamming distance: 3.85
  mean iterations to settle: 9.7
  failures to settle: 0/20
  runs where energy increased: 2/20

Synchronous updating was very fast when it did work, but also failed to settle more times than not.
The network likely continued to flip bac kand forth without converging, as I learned the synchronous updates
can create these types of oscillations. 

Updating 4 nodes at a time created somewhat of a middle ground result. It always settled in the runs as well as 
quickly updating 1 node at a time, but also had times where energy went up as well. It lost some stability in synchronous
updating, where as updating one node at a time was the most stable option. 












"""