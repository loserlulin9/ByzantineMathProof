# Proof for Byzantine Papers

## 1. Krum 

### 1.1 Resource

Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent (NeurIPS 2017)



### 1.2 Model

#### 1.2.1 Threat Model 

Suppose there are $n$ workers, and $f$ of them are Byzantine workers. Each worker $i$ sent gradient vector $V_i$ to the parameter server.

#### 1.2.2 Prerequisites

**(i) (Unbiased expectation)** Let $G$ be the gradient distribution where $V_i \sim G$, we have $\mathbb EG = g$.

**(ii) (Bounded variance)** $\mathbb E ||G-g||^2=d\sigma^2$ where the gradient vectors are $d$-dimensional.

**(iii) (Convex cost function)** The cost function $Q(x)$ needs to be convex.

**(iv) (Extra conditions)** See **Proposition 2** for details.



### 1.3 Defense Method

Krum is to preclude the vectors that are too far away. For any $i \ne j$, we denote by $i \rarr j$ the fact that $V_j$ belongs to the $n-f-2$ closest vectors to $V_i$. Then we define a score $s(i)=\sum_{i \rarr j}||V_i-V_j||^2$ and Krum defense aggregation rule $Krum(V_1,...,V_n)=V_{i_*}$ where $i_*$ refers to the worker which has the lowest score.



**Definition 1.** ($(\alpha, f)-\text{Byzantine Resilience}.$) We say an an aggregation rule $F$ is $(\alpha,f)-\text{Byzantine Resilient}$ if $F$ satisfies:

(i) $\langle \mathbb EF,g \rangle \ge (1-\sin \alpha)·||g||^2 > 0$

(ii) for $r=2,3,4, \mathbb E||F||^r$ is bounded above by a linear combination of terms $\mathbb E||G||^{r_1}...E||G||^{r_{n-1}}$ with $r_1+...+r_{n-1}=r$.



**Proposition 1.** Let $V_1,...,V_n$ be any independent and i.i.d random $d$-dimensional vectors s.t $V_i \sim G$ with $\mathbb EG=g$ and $\mathbb E||G-g||^2=d\sigma^2$. If $2f+2<n$ and $\eta(n,f)\sqrt{d}·\sigma<||g||$ where
$$
\eta(n,f):=\sqrt{2(n-f+\frac{f·(n-f-2)+f^2·(n-f-1)}{n-2f-2})}=\begin{cases} O(n) \enspace \text{if} \enspace f=O(n)\\O(\sqrt{n})\enspace \text{if} f=O(1) \enspace \end{cases}
$$
then the Krum function is $(\alpha,f)-\text{Byzantine Resilient}$ where 
$$
\sin \alpha=\frac{\eta(n,f)·\sqrt d · \sigma}{||g||}
$$



**Proposition 2.** We assume that :

(i) the cost function $Q$ is three times differentiable with continuous derivatives, and is non-negative

(ii) the learning rates satisfy $\sum_t \gamma_t = \infty$ and $\sum_t \gamma _t^2 < \infty$

(iii) $\mathbb E G(x,\xi)=\nabla Q(x)$ and $\forall r \in \{ 2,3,4\}, \mathbb E ||G(x,\xi)||^r \le A_r + B_r||x||^r$ for constants $A_r,B_r$ where $G$ is a gradient estimator

(iv) $\exist \: 0\le \alpha \le \pi/2, \eta(n,f)·\sqrt d · \sigma (x) \le ||\nabla Q(x)|| · \sin \alpha $

(v) beyond a certain horizon, $||x||^2 \ge D$, there exists $\epsilon > 0$ and $0 \le \beta \le \pi/2 - \alpha$ such that $||\nabla Q(x)|| \ge \epsilon > 0$ and $\frac{\langle x, \nabla Q(x)}{||x||·||\nabla Q(x)||} \ge \cos \beta$ 

Then the sequence of gradients $\nabla Q(x_t)$ converges almost surely to zero.



### 1.4 Proof 

#### 1.4.1 For Byzantine Resilient

Consider $Krum=Krum(V_1,...,V_{n-f},B_1,...,B_f)$ and $i_*$ is the index chosen by $Krum$, we have:
$$
\delta_c(i)+\delta_b(i)=n-f-2 \\
n-2f-2 \le \delta_c(i) \le n-f-2 \\
\delta_b(i) \le f
\tag{1-1}
$$
where $\delta_c(i) / \delta_b(i)$ is the number of correct/Byzantine neighbors worker $i$ has.



At first, we focus on the condition (i) of **Definition 1**:
$$
||\mathbb EKrum-g||^2 \le ||\mathbb E (Krum-\frac{1}{\delta_c(i_*)}\sum_{i_* \rarr correct \: j}V_j)||^2 
\\ 
\le \mathbb E||Krum-\frac{1}{\delta_c(i_*)}\sum_{i_* \rarr correct \: j}V_j||^2 \quad \text{(Jensen inequality)} \\
\le \sum_{correct \: j} \mathbb E ||V_i - \frac{1}{\delta_c(i)}\sum_{i \rarr correct \: j}V_j||^2 \mathbb I(i_*=i) 
\\ 
+ \sum_{byz \: k} \mathbb E ||B_k - \frac{1}{\delta_c(k)}\sum_{k \rarr correct \: j}V_j||^2 \mathbb I(i_*=k)
\tag{1-2}
$$
where $\mathbb I$ denotes the indicator function. $\mathbb I(P)=1$ if predicate $P$ is true, and $0$ otherwise.

> **Lemma 1.** (Jensen inequality) If $f(x)$ is convex:
>
> (i) $\mathbb E(f(x)) \ge f(\mathbb E(x))$
>
> (ii) $f(\sum_i \lambda_i x_i) \le \sum_i \lambda_i f(x_i)$, if $\sum_i \lambda_i=1$

When we consider $f(x)=||x||^2$ and $x=Krum-\frac{1}{\delta_c(i_*)}\sum_{i_* \rarr correct \: j}V_j$ in **Lemma 1**-(i), the second inequality sign in equation 1-2 is true.



Continue, we use **Lemma 1**-(ii) and we consider $f(x)=||x||^2$ and $x=V_i-V_j$ in it.
$$
||V_i-\frac{1}{\delta_c(i)}\sum_{i_* \rarr correct \: j}V_j||^2=||\frac{1}{\delta_c(i)}\sum_{i_* \rarr correct \: j}(V_i-V_j)||^2 
\le \frac{1}{\delta_c(i)}\sum_{i_* \rarr correct \: j}||V_i-V_j||^2 \qquad \text{(Jensen inequality)} \\
\mathbb E ||V_i-\frac{1}{\delta_c(i)}\sum_{i \rarr correct \: j} V_j||^2 \le \frac{1}{\delta_c(i)} \sum_{i \rarr correct \: j} \mathbb E ||V_i- V_j||^2 \le 2d\sigma^2 \\
\sum_{correct \: j} \mathbb E ||V_i - \frac{1}{\delta_c(i)}\sum_{i \rarr correct \: j}V_j||^2 \le (n-f)·2d\sigma^2
\tag{1-3}
$$


Now we consider the case where $V_{i_*}=B_k$ is proposed by a Byzantine worker. This represents that $k$ minimizes the score for all indexes $i$ even it is proposed by correct worker:
$$
\sum_{k \rarr correct\:j}||B_k-V_j||^2 + \sum_{k \rarr byz\: l}||B_k-B_l||^2 \le \sum_{i \rarr correct\:j}||V_i-V_j||^2+\sum_{i \rarr byz\:l}||V_i-B_l||^2 
\tag{1-4}
$$


Consider the last term of the equation 1-2, for all indexes $i$ of vectors proposed by correct workers:
$$
||B_k-\frac{1}{\delta_c(k)}\sum_{k \rarr correct\:j}V_j||^2 \le \frac{1}{\delta_c(k)}\sum_{k \rarr correct \: j}||B_k-V_j||^2 \qquad \text{(Jensen inequality)}\\
\le \frac{1}{\delta_c(k)}\sum_{i \rarr correct\:j}||V_i-V_j||^2+\frac{1}{\delta_c(k)}\sum_{i \rarr byz\:l}||V_i-B_l||^2 \qquad \text{(Krum definition)}
\tag{1-5}
$$


We denote $\sum_{i \rarr byz\:l}||V_i-B_l||^2$ as $D^2(i)$ and focus on it:

There exists a correct worker $\zeta(i)$ which is farther from $i$ than every neighbor $j$ of $i$. In particular, for all $l$ such that $i \rarr l$, $||V_i-B_l|| \le ||V_i-V_{\zeta(i)}||^2$. Based on the last inequality of equation 1-5, we have:
$$
||B_k-\frac{1}{\delta_c(k)}\sum_{k \rarr correct\:j}V_j||^2 \le \frac{1}{\delta_c(k)}\sum_{i \rarr correct\:j}||V_i-V_j||^2 + \frac{\delta_b(i)}{\delta_c(k)}||V_i-V_{\zeta(i)}||^2 
\tag{1-6}
$$
For the changing last term, we just replace $D^2(i)$. And we also have:
$$
\mathbb E||B_k-\frac{1}{\delta_c(k)}\sum_{k \rarr correct \: j}V_j||^2 \le \frac{\delta_c(i)}{\delta_c(k)}·2d\sigma^2+\frac{\delta_b(i)}{\delta_c(k)}\sum_{correct \: j \ne i} \mathbb E||V_i-V_j||^2 \mathbb I(\zeta(i)=j) \\
\le (\frac{\delta_c(i)}{\delta_c(k)}+\frac{\delta_b(i)}{\delta_c(k)}(n-f-1))·2d\sigma^2 \\
\le (\frac{n-f-2}{n-2f-2}+\frac{f}{n-2f-2}·(n-f-1))·2d\sigma^2 
\tag{1-7}
$$
For the last inequality, we choose the max of the numerator and the min of the denominator with equation 1-1.

**!!! I have a question for the $(n-f-1)$. Although it is true, it can be reduce to $1$. !!!**



Combining equation 1-2, 1-3, 1-7, we obtain:
$$
||\mathbb E Krum-g||^2 \le [(n-f)+f·(\frac{n-f-2}{n-2f-2}+\frac{f(n-f-1)}{n-2f-2})]·2d\sigma^2 \le \eta^2(n,f)·d\sigma^2
\tag{1-8}
$$


By the assumption of the **Proposition 1**, we have $\eta \sqrt d\sigma < ||g||$, so $\mathbb EKrum$ belongs to a ball centered at $g$ with radius $\eta (n,f) \sqrt d\sigma$. This implies $\langle\mathbb EKrum,g \rangle \ge (1-\sin \alpha)||g||^2$.



We now focus on condition (ii) of the **Definition 1**:
$$
\mathbb E ||Krum||^r=\sum_{correct \: i}\mathbb E||V_i||^r\mathbb I(i_*=i)+\sum_{byz \: k}\mathbb E ||B_k||^r \mathbb I(i_*=k) \le (n-f) \mathbb E ||G||^r + \sum_{byz \: k}\mathbb E||B_k||^r \mathbb I (i_*=k)
\tag{1-9}
$$
When $i_*=k$, for all correct indexes $i$, based on equation 1-6, we have:
$$
||B_k-\frac{1}{\delta_c(k)}\sum_{k \rarr correct\: j}V_j|| \le \sqrt{\frac{1}{\delta_c(k)}\sum_{i \rarr correct \: j}||V_i-V_j||^2+\frac{\delta_b(i)}{\delta_c(k)}||V_i-V_{\zeta(i)}||^2} \\
\le C·(\sqrt {\frac{1}{\delta_c(k)}}·\sum_{i \rarr correct \: j}||V_i-V_j||+\sqrt {\frac {\delta_b(i)}{\delta_c(k)}}||V_i-V_{\zeta(i)}||) \le C·\sum_{correct \: j}||V_j|| \quad \text{(triangular inequality)}
\tag{1-10}
$$
The second inequality comes from the equivalence of norms in finite dimension. Denoting by $C$ a generic constant, we have:
$$
||B_k|| \le ||B_k-\frac{1}{\delta_c(k)}\sum_{k \rarr correct \: j}V_j||+||\frac{1}{\delta_c(k)}\sum_{k \rarr correct \: j}V_j|| \le C·\sum_{correct \: j}||V_j|| \\
||B_k||^r \le C·\sum_{r_1 + ... + r_{n-f}=r}||V_1||^{r_1}···||V_{n-f}||^{r_{n-f}}
\tag{1-11}
$$

Since $V_i$ are independent, we finally obtain that $\mathbb E ||Krum||^r$ is bounded above by a linear combination of the terms because:
$$
\mathbb E||V_1||^{r_1}···\mathbb E||V_{n-f}||^{r_{n-f}}=\mathbb E||G||^{r_1}···\mathbb E||G||^{r_{n-f}}
$$
where $r_1+···+r_{n-f}=r$. And this completes the proof of **Definition 1**-(ii).




#### 1.4.2 For Convergence

The SGD equation is expressed as follows
$$
x_{t+1}=x_t-\gamma_t·Krum(V_1^t,...,V_n^t)=x_t-\gamma_t·Krum_t
\tag{2-1}
$$


We first show that $x_t$ is almost surely globally confined within the region $||x||^2 \le D$. And we let $u_t=\phi(||x_t||^2)$ where
$$
\phi (a)=\begin{cases} \qquad 0 \qquad \text{if} \: a<D \\ (a-D)^2 \quad \text{otherwise} \end{cases} 
$$
And we note that:
$$
\phi(a)-\phi(b)\le (b-a)\phi'(a)+(b-a)^2
\tag{2-2}
$$
this becomes a equality when $a,b\ge D$. Applying this inequality to $u_{t+1}-u_t$ yields:
$$
u_{t+1}-u_t\le (-2\gamma_t \langle x_t,Krum_t\rangle + \gamma_t^2||Krum_t||^2)·\phi'(||x_t||^2)+4\gamma_t^2 \langle x_t,Krum_t\rangle ^2 \\ 
- 4\gamma_t^3 \langle x_t,Krum_t\rangle||Krum_t||^2 + \gamma_t^4||Krum||^4 \\
\le -2 \gamma_t \langle x_t,Krum_t\rangle \phi'(||x_t||^2) + \gamma_t^2||Krum_t||^2 \phi'(||x_t||^2) \\ 
+ 4 \gamma_t^2||x_t||^2||Krum_t||^2 + 4\gamma_t^3||x_t||||Krum_t||^3+\gamma_t^4||Krum_t||^4
\tag{2-3}
$$
where we use $\langle a,b\rangle \le ||a||·||b||$.



Let $P_t$ denote the $\sigma$-algebra encoding all the information up to $t$. We have:
$$
\mathbb E(u_{t+1}-u_t|P_t) \le -2\gamma_t\langle x_t,\mathbb E(Krum_t)\rangle+\gamma_t^2 \mathbb E(||Krum_t||^2)\phi'(||x_t||^2)+4\gamma_t^2||x_t||^2 \mathbb E(||Krum_t||^2) \\ + 4\gamma_t^3||x_t||\mathbb E(||Krum_t||^2) + \gamma_t^4\mathbb E(||Krum_t||^4)
\tag{2-4}
$$


Applying **Definition 1**-(ii) and **Proposition 2**-(iii), we have:
$$
\mathbb E(u_{t+1}-u_t|P_t) \le -2 \gamma_t \langle x_t,\mathbb E(Krum_t)\rangle \phi'(||x_t||^2) + \gamma_t^2(A_0+B_0||x_t||^4) 
\tag{2-5}
$$
Thus, there exists positive constant $A,B$ such that
$$
\mathbb E(u_{t+1}-u_t|P_t) \le -2 \gamma_t \langle x_t,\mathbb E(Krum_t)\rangle \phi'(||x_t||^2) + \gamma_t^2(A+B·u_t) 
\tag{2-6}
$$
And because $\langle x_t,\mathbb E(Krum_t)\rangle \ge ||x_t||·||\mathbb E Krum_t|| ·\cos(\alpha+\beta) > 0$
$$
\mathbb E(u_{t+1}-u_t|P_t) \le \gamma_t^2(A+B·u_t)
\tag{2-7}
$$
**!!! I don't why equation 2-5 is true. Why $\phi'(·)$ could be added to the tail of the right-hand first term based on equation 2-5? !!!**



**!!!----- I can't understand the following proof process. -----!!!**

Then we define two auxiliary sequences:
$$
\mu_t=\prod_{i=1}^{t} \frac{1}{1-\gamma_i^2B} \xrightarrow{t \to \infty} \mu_{\infty} \\
u'_t=\mu_tu_t
\tag{2-8}
$$
Note that $\mu_t$ converges because $\sum_t\gamma_t^2 < \infty$. Then we have:
$$
\mathbb E (u'_{t+1}-u'_t|P_t) \le \gamma^2 \mu_t A
\tag{2-9}
$$
And we define an indicator of the right hand of equation 2-9:
$$
\chi_t=\begin{cases} 1 \qquad \text{if} \: \mathbb E (u'_{t+1}-u'_t|P_t)>0 \\
0 \qquad \text{otherwise}
\end{cases}
\tag{2-10}
$$


Then we have:
$$
\mathbb E(\chi_t·(u'_{t+1}-u'_t)) \le \mathbb E(\chi_t · \mathbb E (u'_{t+1}-u'_t|P_t)) \le \gamma_t^2 \mu_t A
\tag{2-11}
$$
The right-hand side of the previous inequality is the summand of a convergent series. By the quasi-martingale convergence theorem, this shows that the sequence $u'_t$ converges almost surely, which in turn shows that the sequence $u_t$ converges almost surely, $u_t\to u_{\infty} \ge 0$.



Let us assume $u_{\infty}>0$. When $t$ is large enough, this implies that $||x_t||^2,||x_{t+1}||^2>D$ and equation 2-2 becomes an equality which implies that the following infinite sum converges almost surely:
$$
\sum_{t=1}^{\infty}\gamma_t \langle x_t,\mathbb EKrum_t \rangle \phi'(||x_t||^2) < \infty
\tag{2-12}
$$


Note that the sequence $\phi'(||x_t||^2)$ converges to a positive value. In the region $||x_t||^2>D$, we have:
$$
\langle x_t,\mathbb E Krum_t \rangle \ge \sqrt D· ||\mathbb EKrum_t|| · \cos (\alpha + \beta) \\
\ge \sqrt D· (||\nabla Q(x_t)-\eta(n,f)·\sqrt d·\sigma(x_t)||) · \cos (\alpha + \beta) \\
\ge \sqrt D· \epsilon· (1-\sin \alpha) · \cos (\alpha + \beta) > 0
\tag{2-13}
$$
This contradicts the fact that $\sum_{t=1}^{\infty} \gamma_t=\infty$. Therefore, the sequence $u_t$ converges to zero. This convergence implies that the sequence $||x_t||^2$ is bounded. 

As a sequence, any continuous function of $x_t$ is also bounded, such as $||x_t||^2,\mathbb E||G(x,\xi)||^2$ and all the derivatives of the cost function $Q(x_t)$. And we will use $K_i$ as a positive constant whenever such a bound is used.

**!!!----- I can't understand the proof process above. -----!!!**



We proceed to show that the gradient $\nabla Q(x_t) = \nabla h_t$ converges almost to zero.

Using Taylor expansion and bounding the second derivative with $K_1$, we obtain:
$$
|h_{t+1}-h_t+2\gamma_t \langle Krum_t,\nabla Q(x_t) \rangle| \le \gamma_t^2 ||Krum_t||^2 K_1
\tag{2-14}
$$
Therefore,
$$
\mathbb E(h_{t+1}-h_t|P_t) \le -2\gamma_t \langle Krum_t,\nabla Q(x_t) \rangle + \gamma_t^2 \mathbb E(||Krum_t||^2|P_t)K_1
\tag{2-15}
$$
Under **Definition 1**, this implies:
$$
\mathbb E(h_{t+1}-h_t|P_t) \le \gamma_t^2K_2K_1
\tag{2-16}
$$
which in turn implies:
$$
\mathbb E(\chi_t ·(h_{t+1}-h_t)) \le \gamma_t^2K_2K_1
\tag{2-17}
$$
The right-hand side is the summand of a convergent infinite sum. By the quasi-martingale convergence theorem, the sequence $h_t$ converges almost surely, $Q(x_t) \to Q_{\infty}$.



Taking the expectation of equation 2-15, and computing the sum from $t=1$, the convergence of $Q(x_t)$ implies that
$$
\sum_{t=1}^{\infty}\gamma_t \langle \mathbb EKrum_t,\nabla Q(x_t) \rangle < \infty
\tag{2-18}
$$
Using a Taylor expansion and defining $\rho_t=||\nabla Q(x_t)||^2$, we obtain:
$$
\rho_{t+1}-\rho_t \le -2 \gamma_t \langle Krum_t,(\nabla^2Q(x_t))·\nabla Q(x_t) \rangle + \gamma_t^2 ||Krum_t||^2K_3
\tag{2-19}
$$


Taking the conditional expectations, and bounding the second derivatives by $K_4$
$$
\mathbb E (\rho_{t+1}-\rho_t|P_t) \le 2\gamma_t \langle \mathbb EKrum_t,\nabla Q(x_t) \rangle K_4 + \gamma_t^2K_2K_3
\tag{2-20}
$$


The positive expected variations of $\rho_t$ are bounded
$$
\mathbb E(\chi_t·(\rho_{t+1}-\rho_t)) \le 2\gamma_t \mathbb E\langle \mathbb EKrum_t,\nabla Q(x_t) \rangle K_4 + \gamma_t^2K_2K_3
\tag{2-21}
$$
The two terms on the right-hand side are the summands of convergent infinite series. By the quasi-martingale convergence theorem, this shows that $\rho_t$ converges almost surely.



We have
$$
\langle \mathbb EKrum_t,\nabla Q(x_t) \rangle \ge (||\nabla Q(x_t)|| - \eta(n,f)·\sqrt d · \sigma(x_t)) · ||\nabla Q(x_t)|| \ge (1-\sin \alpha) · \rho_t
\tag{2-22}
$$
This implies that the following infinite series converge almost surely:
$$
\sum_{t=1}^{\infty} \gamma_t · \rho_t < \infty
\tag{2-23}
$$


Since $\rho_t$ converges almost surely, and series $\sum_{t=1}^{\infty} \gamma_t =\infty$ diverges, we conclude that the sequence $||\nabla Q(x_t)||$ converges almost surely to zero.




### 1.5 Results

**Result 1.** A single Byzantine worker can prevent the convergence a linear aggregation rule.

**Result 2.** The expected time complexity of the Krum Function is $O(n^2·d)$ where gradient vectors are $d$-dimensional.

**Result 3.** Krum is Byzantine Resilient.

**Result 4.** By using Krum, $\nabla Q(x_t)$ converges almost surely to zero.













