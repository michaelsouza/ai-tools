equation[[178, 153, 872, 316]]
 \[ \begin{align*}&\int_{\tau_{1}}^{\tau_{2}}\sqrt{\tau}(<Y,\nabla R>+2<\nabla_{Y}X,X>)d\tau=\int_{\tau_{1}}^{\tau _{2}}\sqrt{\tau}(< Y,\nabla R>+2<\bigtriangledown_{X}Y,X>)d\tau\\&\quad=\int_{\tau_{1}}^{ \tau_{2}}\sqrt{\tau}(<Y,\nabla R>+2\frac{d}{d\tau}<Y,X>-2<Y,\bigtriangledown_{X}X>-4\mathrm{Ric}(Y,X))d\tau\\&\quad=2\sqrt{\tau}<X,Y>\big|_{\tau_{1}}^{\tau_{2}}+\int_{\tau_{1}}^{ \tau_{2}}\sqrt{\tau}<Y,\nabla R-2\bigtriangledown_{X}X-4\mathrm{Ric}(X,\cdot)-\frac{1}{\tau}X>d\tau\end{align*} \quad (7.1) \] 

text[[177, 331, 434, 348]]
Thus L-geodesics must satisfy

equation[[333, 360, 816, 396]]
 \[ \nabla_{X}X-\frac{1}{2}\nabla R+\frac{1}{2\tau}X+2\mathrm{R i c}(X,\cdot)=0 \quad (7.2) \] 

text[[176, 407, 818, 570]]
Given two points p, q and  \( \tau_{2} > \tau_{1} > 0 \) , we can always find an L-shortest curve  \( \gamma(\tau) \) ,  \( \tau \in [\tau_{1}, \tau_{2}] \)  between them, and every such L-shortest curve is L-geodesic. It is easy to extend this to the case  \( \tau_{1} = 0 \) ; in this case  \( \sqrt{\tau} X(\tau) \)  has a limit as  \( \tau \to 0 \) . From now on we fix p and  \( \tau_{1} \approx 0 \)  and denote by  \( L(q, \bar{\tau}) \)  the L-length of the L-shortest curve  \( \gamma(\tau) \) ,  \( 0 \leq \tau \leq \bar{\tau} \) , connecting p and q. In the computations below we pretend that shortest L-geodesics between p and q are unique for all pairs  \( (q, \bar{\tau}) \) ; if this is not the case, the inequalities that we obtain are still valid when understood in the barrier sense, or in the sense of distributions.

text[[176, 571, 817, 608]]
The first variation formula (7.1) implies that  \( \nabla L(q,\bar{\tau}) = 2\sqrt{\bar{\tau}}X(\bar{\tau}) \) , so that  \( |\nabla L|^{2} = 4\bar{\tau}|X|^{2} = -4\bar{\tau}R + 4\bar{\tau}(R + |X|^{2}) \) . We can also compute

equation[[227, 621, 767, 641]]
 \[ L_{\bar{\tau}}(q,\bar{\tau})=\sqrt{\bar{\tau}}(R+|X|^{2})-<X,\nabla L>=2\sqrt{\bar{\tau}}R-\sqrt{\bar{\tau}}(R+|\dot{X}|^{2}) \] 

text[[177, 656, 576, 674]]
To evaluate  \( R + |X|^{2} \)  we compute (using (7.2))

equation[[184, 687, 815, 824]]
 \[ \begin{align*}\frac{d}{d\tau}(R(\gamma(\tau))+|X(\tau)|^{2})&=R_{\tau}+<\nabla R,X>+2<\nabla_{X}X,X>+2\mathrm{Ric}(X,X)\\&=R_{\tau}+\frac{1}{\tau}R+2<\nabla R,X>-2\mathrm{Ric}(X,X)-\frac{1}{\tau}(R+|X|^{2})\\&=-H(X)-\frac{1}{\tau}\big(R+|X|^{2}\big),\end{align*} \quad (7.3) \]