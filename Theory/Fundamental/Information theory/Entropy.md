# Entropy

## Annotation

### **1. $p_i$: Probability**

* **What it is**: The probability of a specific event/outcome.
* **Meaning**: How likely is one particular outcome.
* **Type**: Number between 0 and 1.
* **Example**:
  If $X$ is a die roll,
  $p_3 = P(X = 3) = \frac{1}{6}$

### **2. $H(\cdot)$: Entropy**

* **What it is**: A function of the entire **distribution** of a random variable.
* **Meaning**: Measures the **total uncertainty** across all possible outcomes.
* **Type**: Non-negative real number (in bits or nats).
* **Example**:
  If $X$ is a fair coin, then:

  $$
  H(X) = -\left( \frac{1}{2} \log_2 \frac{1}{2} + \frac{1}{2} \log_2 \frac{1}{2} \right) = 1 \text{ bit}
  $$

### **Relationship**

Entropy is a **summary of the uncertainty** across all $p_i$:

$$
H(X) = -\sum_i p_i \log p_i
$$

* Each $p_i$ contributes to $H(X)$
* If all $p_i$ are equal (uniform distribution), entropy is **maximal**
* If one $p_i = 1$ and the rest are 0, entropy is **zero** (no uncertainty)

### **Summary Table**

| Symbol | Stands for               | Represents              | Depends on       |
| ------ | ------------------------ | ----------------------- | ---------------- |
| $p_i$  | Probability of event $i$ | Likelihood of one value | One event        |
| $H(X)$ | Entropy of variable $X$  | Total uncertainty       | All $p_i$ values |

---

## **Additivity of Independent Events**

### **What it means:**

If you have two **independent random variables**, the total uncertainty about both should be the **sum** of the individual uncertainties.

Formally:
If $X$ and $Y$ are independent, then:

$$
H(X, Y) = H(X) + H(Y)
$$

This principle is called **additivity of entropy**.

### **Why this makes sense:**

Imagine you’re playing a guessing game. The more possible outcomes, the harder it is to guess, and the more questions you need to ask.

Now, let's look at some examples.

### **Example 1: Coin Toss + Die Roll**

Let’s define:

* $X$: fair coin → {Heads, Tails} → 2 outcomes
* $Y$: fair die → {1, 2, 3, 4, 5, 6} → 6 outcomes

#### **How many outcomes are possible together?**

If both happen **independently**, you get:

* Coin × Die = 2 × 6 = 12 total outcomes

So to describe **both events together**, you need to distinguish among 12 combinations like:

* (Heads, 1), (Heads, 2), … (Tails, 6)

#### **Why is entropy additive here?**

Because:

* Describing the coin needs a certain number of bits (or amount of uncertainty)
* Describing the die needs more
* Describing both together = sum of both amounts

You’re **not repeating effort**. You can first guess the coin, then the die. Each step adds information but doesn’t overlap, because the events are independent.

### **Example 2: 2 Independent Coins**

Now take two fair coins:

* $A$: {H, T}
* $B$: {H, T}

They are independent. So the joint event space is:

* (H, H), (H, T), (T, H), (T, T) → 4 outcomes

Entropy of each coin:

* $H(A) = \text{some amount}$
* $H(B) = \text{same amount}$
* $H(A, B) = H(A) + H(B)$

Because: guessing both is like playing two rounds of the same game. No shortcuts.

### **Why it must be additive (intuition)**

If uncertainty didn’t add, then knowing one event would somehow tell you something about the other — which violates **independence**.

So:

* Independent → no overlap in information
* Therefore, **total uncertainty** = **sum** of uncertainties

---

## **Goal of Entropy**

Entropy tries to **measure information or uncertainty**.

To do that, we must ask:
**How much information do I get when I see an outcome?**

Answer: depends on **how surprising** the outcome is.

### **Step 1: Define "information from an outcome"**

We define:

$$
I(p_i) = \text{information content of outcome } i
$$

And we want a few rules to hold:

#### A. Rare outcomes give **more** information

* $I(0.01) > I(0.5)$
  You’re more surprised by something unlikely.

#### B. Independent outcomes should **add** their information

* If you roll a die **and** flip a coin:

  $$
  I(p_1 \cdot p_2) = I(p_1) + I(p_2)
  $$

Only **logarithms** satisfy this:

$$
\log(p_1 \cdot p_2) = \log(p_1) + \log(p_2)
$$

So, we define:

$$
I(p_i) = -\log_b(p_i)
$$

The minus sign is there because $\log(p_i)$ is negative when $p_i < 1$, and we want information to be **positive**.

### **Step 2: Entropy = Expected Information**

Entropy is the **average surprise** across all possible outcomes:

$$
H(X) = \sum_{i=1}^{n} p_i \cdot I(p_i) = -\sum_{i=1}^{n} p_i \log_b p_i
$$

So the **logarithm tells us how much one outcome "says"**, and entropy is the average of all these.

### **Intuition with Examples**

#### 1. Uniform distribution: maximum entropy

* Toss a fair coin:

  $$
  H = -\left( \frac{1}{2} \log_2 \frac{1}{2} + \frac{1}{2} \log_2 \frac{1}{2} \right) = 1 \text{ bit}
  $$

Every outcome is equally surprising. Maximum uncertainty.

#### 2. Skewed distribution: less entropy

* Toss a biased coin:

  $$
  P(H) = 0.99,\quad P(T) = 0.01
  $$

  $$
  H \approx - (0.99 \cdot \log_2 0.99 + 0.01 \cdot \log_2 0.01) \approx 0.08 \text{ bits}
  $$

Almost no uncertainty. You almost always get Heads.

### **If there were no log...?**

Try just using:

$$
H(X) = \sum p_i (1 - p_i) \text{ or } p_i^2
$$

You would break:

* Additivity: $H(X, Y) \neq H(X) + H(Y)$
* Rare events wouldn’t be more informative
* Max entropy wouldn't be at uniform distribution

Only **log** preserves all required properties.

## Bit

A **bit** (short for *binary digit*) is the **smallest unit of information**. It can represent **two possibilities**:

$$
\text{bit} \in \{0, 1\}
$$

### **What does "1 bit of information" mean?**

It means you are able to **distinguish between two equally likely outcomes**.

Examples:

* Coin toss (Heads or Tails) → 1 bit
* Answering a yes/no question → 1 bit

### **How bits relate to uncertainty**

If you want to identify:

* One of **4** equally likely outcomes → need **2 bits**:
  $2^2 = 4$
* One of **8** outcomes → need **3 bits**:
  $2^3 = 8$

In general:

$$
\text{number of bits needed} = \log_2(\text{number of possible outcomes})
$$

But only if all outcomes are **equally likely** and you use optimal encoding.

### **Bits in Entropy**

In information theory:

$$
H(X) = \text{Expected number of bits to encode outcome of } X
$$

If:

* $H(X) = 1$ bit → you're as uncertain as when flipping a fair coin.
* $H(X) = 3$ bits → it's like guessing 1 value out of 8 options.

### **Real-World Analogy**

Think of 1 bit as answering a **yes/no** question:

* “Is it raining?” → 1 bit
* “Is the number greater than 10?” → 1 bit

You can chain them:

* 3 bits = 3 yes/no questions → can distinguish among $2^3 = 8$ options

That’s how bits **quantify information**.

---

## **why** we define

$$
I(p_i) = -\log_b p_i
$$

Even though **I** is a different function name than **log**, the definition uses the logarithm. Here’s why:

### **We want information to behave in a specific way**

Let’s say:

* $p_i$: probability of an outcome
* $I(p_i)$: information (surprise) of that outcome

We want $I(p)$ to satisfy the following **natural requirements**:

### **1. Information should decrease when probability increases**

* If an event is more likely, it should be **less surprising**
* Mathematically: $p_1 > p_2 \Rightarrow I(p_1) < I(p_2)$

### **2. Independent events should add information**

If you have two **independent events**:

* Event A with probability $p$
* Event B with probability $q$

Then joint probability = $p \cdot q$

We want:

$$
I(p \cdot q) = I(p) + I(q)
$$

Only **logarithms** satisfy this property:

$$
\log(p \cdot q) = \log(p) + \log(q)
$$

So we define:

$$
I(p) = -\log_b(p)
$$

The minus sign ensures that the result is **positive**, since $\log(p) < 0$ when $0 < p < 1$

### **3. Normalization with base $b$**

* If we use base 2 → information is measured in **bits**
* If base $e$ → information is in **nats**
* If base 10 → it's in **Hartleys**

### **Conclusion**

We define:

$$
I(p_i) := -\log_b(p_i)
$$

because:

* It satisfies the intuitive properties of surprise
* It’s additive for independent events
* It scales correctly using logarithmic base

It’s not that $I$ *is* the log — it’s that log is the only function satisfying the constraints we want $I$ to obey.

---

## **1. Information from One Outcome**

We define:

$$
I(p_i) = \text{information from seeing outcome } i = -\log_b(p_i)
$$

Why?

* The less likely something is, the **more surprising** it is.
* Example: If $p_i = 0.5$, then $I(p_i) = 1$ bit.
  If $p_i = 0.01$, then $I(p_i) = 6.64$ bits.

The logarithm captures how **unexpected** an outcome is.

---

### **2. Entropy: Expected (Average) Surprise**

Now, suppose you have a random variable $X$ with probabilities:

$$
P(x_1) = p_1, \quad P(x_2) = p_2, \quad \ldots, \quad P(x_n) = p_n
$$

Each outcome gives you $I(p_i)$ bits of info when it happens.

But not all outcomes are equally likely. So the **average information** you get is:

$$
H(X) = \sum_{i=1}^n p_i \cdot I(p_i) = -\sum_{i=1}^n p_i \log_b(p_i)
$$

That’s entropy.

It’s like saying:

> "Let’s multiply the info of each possible event by how often it happens, then sum it."

### **Concrete Example**

Let’s say you have this distribution:

$$
P(A) = 0.5,\quad P(B) = 0.25,\quad P(C) = 0.25
$$

Now compute the information for each outcome:

* $I(A) = -\log_2(0.5) = 1$
* $I(B) = I(C) = -\log_2(0.25) = 2$

Then the entropy is:

$$
H(X) = 0.5 \cdot 1 + 0.25 \cdot 2 + 0.25 \cdot 2 = 1.5 \text{ bits}
$$

**Interpretation**:

* On average, you learn **1.5 bits** per observation of $X$
* Rare outcomes (like B, C) are more surprising — 2 bits — but they happen less often
* The average surprise balances this out

### **Intuition**

Think of a quiz game:

* If the answer is always obvious (like $p_i = 1$), then surprise is 0.
* If answers are uncertain, you get more surprise per question.
* Entropy is the **average surprise per question**, based on how often each answer shows up.

---

## Examples

Here are concrete entropy results for different probability distributions with interpretations. We use **base 2** (so entropy is measured in **bits**) and assume outcomes $x_1, x_2, ..., x_n$.

### **1. Fair Coin Toss**

$$
P(H) = 0.5,\quad P(T) = 0.5
$$

$$
H = -[0.5 \log_2 0.5 + 0.5 \log_2 0.5] = -2 \cdot 0.5 \cdot (-1) = 1 \text{ bit}
$$

**Interpretation**:

* Maximum uncertainty for 2 outcomes.
* On average, you need **1 bit** to describe the result.
* Every toss is equally surprising.

### **2. Biased Coin**

$$
P(H) = 0.9,\quad P(T) = 0.1
$$

$$
H = -[0.9 \log_2 0.9 + 0.1 \log_2 0.1] \approx -[0.9 \cdot (-0.152) + 0.1 \cdot (-3.32)] \approx 0.47 \text{ bits}
$$

**Interpretation**:

* Less uncertainty.
* On average, less than 1 bit needed: most of the time it’s heads, so outcome is not very surprising.

### **3. Fair 4-Sided Die**

$$
P(i) = 0.25 \text{ for } i = 1,2,3,4
$$

$$
H = -4 \cdot 0.25 \cdot \log_2 0.25 = -4 \cdot 0.25 \cdot (-2) = 2 \text{ bits}
$$

**Interpretation**:

* You need **2 bits** to encode the outcome (00, 01, 10, 11).
* This is the max for 4 outcomes.

### **4. Uneven Distribution Over 4 Outcomes**

$$
P = (0.7, 0.1, 0.1, 0.1)
$$

$$
H \approx -[0.7 \cdot \log_2 0.7 + 3 \cdot 0.1 \cdot \log_2 0.1] \approx -[0.7 \cdot (-0.515) + 3 \cdot 0.1 \cdot (-3.32)] \approx 1.36 \text{ bits}
$$

**Interpretation**:

* Dominant outcome (0.7) reduces overall uncertainty.
* You need **less than 2 bits** on average to describe outcomes.

### **5. One Certain Outcome**

$$
P = (1.0, 0.0, 0.0, 0.0)
$$

$$
H = -[1 \cdot \log_2 1 + 0] = 0 \text{ bits}
$$

**Interpretation**:

* No uncertainty at all.
* You don’t need to store or transmit anything to describe the outcome.

### **Summary Table**

| Distribution             | Entropy (bits) | Interpretation     |
| ------------------------ | -------------- | ------------------ |
| (0.5, 0.5)               | 1.00           | Max for 2 outcomes |
| (0.9, 0.1)               | 0.47           | Low uncertainty    |
| (0.25, 0.25, 0.25, 0.25) | 2.00           | Max for 4 outcomes |
| (0.7, 0.1, 0.1, 0.1)     | 1.36           | Medium uncertainty |
| (1.0, 0.0, 0.0, 0.0)     | 0.00           | No uncertainty     |

---

**Q: How entropy formula works? Why is there a log in it?**

A:
Entropy measures average uncertainty or information.

* The log appears because we want information from independent events to add:

  $$
  I(p_1 \cdot p_2) = I(p_1) + I(p_2)
  $$
* Only the logarithm satisfies this property:

  $$
  \log(p_1 \cdot p_2) = \log(p_1) + \log(p_2)
  $$
* We define information content of outcome $i$ as:

  $$
  I(p_i) = -\log_b p_i
  $$
* Entropy is the average information:

  $$
  H(X) = -\sum p_i \log_b p_i
  $$

---

**Q: Explain Additivity of Independent Events with examples.**

A:

* If two events $A$ and $B$ are independent, then:

  $$
  P(A \text{ and } B) = P(A) \cdot P(B)
  $$
* Information content adds:

  $$
  I(P(A \text{ and } B)) = I(P(A)) + I(P(B))
  $$
* Because $I(p) = -\log p$,

  $$
  -\log (P(A) \cdot P(B)) = -\log P(A) - \log P(B)
  $$

Example:

* Toss a fair coin (p=0.5) and roll a fair die (p=1/6).
* Info from coin: 1 bit, info from die: about 2.58 bits
* Combined info: 3.58 bits = 1 + 2.58 bits

---

**Q: What is the difference between $H(\text{something})$ and $p_i$?**

A:

* $p_i$ is the probability of the $i$-th outcome (a number between 0 and 1).
* $H$ is entropy, a single number summarizing the average information across all $p_i$.
* $H = -\sum p_i \log p_i$ depends on the whole probability distribution.

---

**Q: What is the role of the log function in entropy?**

A:

* Logarithm transforms probabilities into additive information values.
* It ensures that:

  * Rare events give more information.
  * Information from independent events sums.
* Without log, these properties fail.

---

**Q: Show some entropy results and interpret them**

A:

| Distribution                        | Entropy (bits) | Interpretation                     |
| ----------------------------------- | -------------- | ---------------------------------- |
| Fair coin (0.5, 0.5)                | 1.00           | Maximum uncertainty for 2 outcomes |
| Biased coin (0.9, 0.1)              | 0.47           | Less uncertainty                   |
| Fair 4-sided die (0.25 each)        | 2.00           | Maximum uncertainty for 4 outcomes |
| Uneven 4 outcomes (0.7,0.1,0.1,0.1) | 1.36           | Some outcomes more likely          |
| Certain outcome (1.0, 0.0...)       | 0.00           | No uncertainty                     |

---

**Q: For Fair 4-Sided Die you need only 2 bits? I think you need 3 bits.**

A:

* You need **2 bits** in fixed-length binary coding, because:

  $$
  \lceil \log_2 4 \rceil = 2
  $$
* 3 bits would allow up to 8 outcomes, which is more than needed.
* Entropy (average info) also equals 2 bits here, meaning optimal coding needs 2 bits on average.

---

**Q: What do bits mean?**

A:

* A bit is the smallest unit of information: 0 or 1.
* 1 bit distinguishes between two equally likely outcomes.
* Number of bits needed to represent $n$ outcomes is about $\log_2 n$ if equally likely.
* In entropy, bits measure the average number of yes/no questions needed to identify the outcome.

---

**Q: Explain why entropy is average surprise:**

A:

* Each outcome $i$ has information:

  $$
  I(p_i) = -\log p_i
  $$
* The entropy averages this weighted by how often each outcome occurs:

  $$
  H = \sum p_i I(p_i) = -\sum p_i \log p_i
  $$
* Example: For $P = (0.5, 0.25, 0.25)$:

  $$
  I(0.5) = 1, \quad I(0.25) = 2
  $$

  $$
  H = 0.5 \times 1 + 0.25 \times 2 + 0.25 \times 2 = 1.5 \text{ bits}
  $$
* This means on average you get 1.5 bits of information per observation.
