
$$\int_{\tau_1}^{\tau_2} \sqrt{\tau} (<Y, \nabla R > +2 < \nabla_Y X, X >) d\tau = \int_{\tau_1}^{\tau_2} \sqrt{\tau} (<Y, \nabla R > +2 < \nabla_X Y, X >) d\tau$$
$$= \int_{\tau_1}^{\tau_2} \sqrt{\tau} (<Y, \nabla R > +2 \frac{d}{d\tau} < Y, X > -2 < Y, \nabla_X X > -4 \text{Ric}(Y, X)) d\tau$$
$$= 2 \sqrt{\tau} < X, Y > \big|_{\tau_1}^{\tau_2} + \int_{\tau_1}^{\tau_2} \sqrt{\tau} < Y, \nabla R - 2 \nabla_X X - 4 \text{Ric}(X, \cdot) - \frac{1}{\tau} X > d\tau$$

(7.1)

Thus $\mathcal{L}$-geodesics must satisfy

$$\nabla_X X - \frac{1}{2} \nabla R + \frac{1}{2 \tau} X + 2 \text{Ric}(X, \cdot) = 0$$

(7.2)

Given two points $p, q$ and $\tau_2 > \tau_1 > 0$, we can always find an $\mathcal{L}$-shortest curve $\gamma(\tau), \tau \in [\tau_1, \tau_2]$ between them, and every such $\mathcal{L}$-shortest curve is $\mathcal{L}$-geodesic. It is easy to extend this to the case $\tau_1 = 0$; in this case $\sqrt{\tau} X(\tau)$ has a limit as $\tau \to 0$. From now on we fix $p$ and $\tau_1 = 0$ and denote by $L(q, \bar{\tau})$ the $\mathcal{L}$-length of the $\mathcal{L}$-shortest curve $\gamma(\tau), 0 \leq \tau \leq \bar{\tau}$, connecting $p$ and $q$. In the computations below we pretend that shortest $\mathcal{L}$-geodesics between $p$ and $q$ are unique for all pairs $(q, \bar{\tau})$; if this is not the case, the inequalities that we obtain are still valid when understood in the barrier sense, or in the sense of distributions.

The first variation formula (7.1) implies that $\nabla L(q, \bar{\tau}) = 2 \sqrt{\tau} X(\bar{\tau})$, so that $|\nabla L|^2 = 4 \bar{\tau}|X|^2 = -4 \bar{\tau} R + 4 \bar{\tau}(R + |X|^2)$. We can also compute

$$L_{\bar{\tau}}(q, \bar{\tau}) = \sqrt{\bar{\tau}}(R + |X|^2) - <X, \nabla L > = 2 \sqrt{\bar{\tau}} R - \sqrt{\bar{\tau}}(R + |X|^2)$$

To evaluate $R + |X|^2$ we compute (using (7.2))

$$\frac{d}{d\tau} \left( R(\gamma(\tau)) + |X(\tau)|^2 \right) = R_{\tau} + <\nabla R, X > +2 < \nabla_X X, X > +2 \text{Ric}(X, X)$$
$$= R_{\tau} + \frac{1}{\tau} R + 2 < \nabla R, X > -2 \text{Ric}(X, X) - \frac{1}{\tau}(R + |X|^2)$$
$$= -H(X) - \frac{1}{\tau}(R + |X|^2),$$

(7.3)