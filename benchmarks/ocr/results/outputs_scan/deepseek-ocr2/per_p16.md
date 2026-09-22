text[[177, 161, 808, 200]]
where  \( H(X) \)  is the Hamilton's expression for the trace Harnack inequality (with  \( t = -\tau \) ). Hence,

equation[[342, 208, 806, 243]]
 \[ \bar{\tau}^{\frac{3}{2}}(R+|X|^{2})(\bar{\tau})=-K+\frac{1}{2}L(q,\bar{\tau}), \quad (7.4) \] 

text[[178, 254, 808, 294]]
where  \(  K = K(\gamma, \bar{\tau})  \)  denotes the integral  \(  \int_{0}^{\bar{\tau}} \tau^{\frac{3}{2}} H(X) d\tau  \) , which we'll encounter a few times below. Thus we get

equation[[385, 302, 807, 337]]
 \[ L_{\bar{\tau}}=2\sqrt{\bar{\tau}}R-\frac{1}{2\bar{\tau}}L+\frac{1}{\bar{\tau}}K \quad (7.5) \] 

equation[[362, 358, 808, 396]]
 \[ |\nabla L|^{2}=-4\bar{\tau}R+\frac{2}{\sqrt{\bar{\tau}}}L-\frac{4}{\sqrt{\bar{\tau}}K} \quad (7.6) \] 

text[[210, 406, 770, 425]]
Finally we need to estimate the second variation of L. We compute

equation[[182, 435, 840, 529]]
 \[ \begin{align*}\delta_{Y}^{2}(\mathcal{L})&=\int_{0}^{\bar{\tau}}\sqrt{\tau}(Y\cdot Y\cdot R+2<\nabla_{Y}\nabla_{Y}X,X>+2|\nabla_{Y}X|^{2})d\tau\\&=\int_{0}^{\tau}\sqrt{\tau}(Y\cdot Y\dot{\cdot}R+2<\nabla_{X}\nabla_{Y}Y,X>+2<R(Y,X),Y,X>+2|\nabla_{X}Y|^{2})d\tau\end{align*} \] 

text[[183, 531, 226, 547]]
Now

equation[[184, 556, 877, 590]]
 \[ \frac{d}{d\tau}<\nabla_{Y}Y,X>=<\nabla_{X}\nabla_{Y}Y,X>+<\nabla_{Y}Y,\nabla_{X}X>+2Y\cdot\mathrm{R i c}(Y,X)-X\cdot\mathrm{R i c}(Y,Y), \] 

text[[183, 600, 353, 617]]
so, if  \( Y(0)=0 \)  then

equation[[375, 626, 623, 647]]
 \[ \delta_{Y}^{2}(\mathcal{L})=2<\nabla_{Y}Y,X>\sqrt{\bar{\tau}}+ \] 

equation[[204, 676, 816, 733]]
 \[ \begin{align*}\int_{0}^{\bar{\tau}}\sqrt{\tau}(\nabla_{Y}\nabla_{Y}R+2<R(Y,X),Y,X>+2|\nabla_{X}Y|^{2}\\+2\nabla_{X}\mathrm{Ric}(Y,Y)-4\nabla_{Y}\mathrm{Ric}(Y,X))d\tau,\end{align*} \quad (7.7) \] 

text[[186, 745, 818, 802]]
where we discarded the scalar product of  \( -2\nabla_{Y}Y \)  with the left hand side of (7.2). Now fix the value of Y at  \( \tau=\bar{\tau} \) , assuming  \( |Y(\bar{\tau})|=1 \) , and construct Y on  \( [0,\bar{\tau}] \)  by solving the ODE

equation[[391, 810, 817, 846]]
 \[ \nabla_{X}Y=-\mathrm{R i c}(Y,\cdot)+\frac{1}{2\tau}Y \quad (7.8) \]