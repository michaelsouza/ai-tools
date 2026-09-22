text[[176, 160, 818, 196]]
where  \( H(X) \)  is the Hamilton's expression for the trace Harnack inequality (with  \( t = -\tau \) ). Hence,

equation[[345, 207, 816, 240]]
 \[ \bar{\tau}^{\frac{3}{2}}(R+|X|^{2})(\bar{\tau})=-K+\frac{1}{2}L(q,\bar{\tau}), \quad (7.4) \] 

text[[175, 254, 818, 291]]
where  \(  K = K(\gamma, \bar{\tau})  \)  denotes the integral  \(  \int_{0}^{\bar{\tau}} \tau^{\frac{3}{2}} H(X) d\tau  \) , which we'll encounter a few times below. Thus we get

equation[[385, 301, 816, 336]]
 \[ L_{\bar{\tau}}=2\sqrt{\bar{\tau}}R-\frac{1}{2\bar{\tau}}L+\frac{1}{\bar{\tau}}K \quad (7.5) \] 

equation[[364, 358, 816, 396]]
 \[ |\nabla L|^{2}=-4\bar{\tau}R+\frac{2}{\sqrt{\bar{\tau}}}L-\frac{4}{\sqrt{\bar{\tau}}K} \quad (7.6) \] 

text[[206, 406, 776, 423]]
Finally we need to estimate the second variation of L. We compute

equation[[176, 434, 851, 544]]
 \[ \begin{align*}\delta_{Y}^{2}(\mathcal{L})=\int_{0}^{\bar{\tau}}\sqrt{\tau}(Y\cdot Y\cdot R+2<\nabla_{Y}\nabla_{Y}X,X>+2|\nabla_{Y}X|^{2})d\tau\\=\int_{0}^{\bar{\tau}}&\sqrt{\tau}(Y\cdot Y \cdot R+2<\nabla_{X}\nabla_{Y}Y,X>+2<R(Y,X),Y,X>+2|\nabla_{X}Y|^{2})d\tau\end{align*} \] 

text[[177, 529, 220, 544]]
Now

equation[[176, 555, 883, 591]]
 \[ \frac{d}{d\tau}<\nabla_{Y}Y,X>=<\nabla_{X}\nabla_{Y}Y,X>+<\nabla_{Y}Y,\nabla_{X}X>+2Y\cdot\mathrm{R i c}(Y,X)-X\cdot\mathrm{R i c}(Y,Y), \] 

text[[176, 599, 349, 617]]
so, if  \( Y(0)=0 \)  then

equation[[371, 628, 622, 648]]
 \[ \delta_{Y}^{2}(\mathcal{L})=2<\nabla_{Y}Y,X>\sqrt{\bar{\tau}}+ \] 

equation[[196, 675, 816, 736]]
 \[ \begin{align*}\int_{0}^{\bar{\tau}}\sqrt{\tau}(\nabla_{Y}\nabla_{Y}R+2<R(Y,X),Y,X>+2|\nabla_{X}Y|^{2}\\+2\nabla_{X}\mathrm{Ric}(Y,Y)-4\nabla_{Y}\mathrm{Ric}(Y,X))d\tau,\end{align*} \quad (7.7) \] 

text[[175, 749, 819, 804]]
where we discarded the scalar product of  \( -2\nabla_{Y}Y \)  with the left hand side of (7.2). Now fix the value of Y at  \( \tau=\bar{\tau} \) , assuming  \( |Y(\bar{\tau})|=1 \) , and construct Y on  \( [0,\bar{\tau}] \)  by solving the ODE

equation[[384, 813, 816, 848]]
 \[ \nabla_{X}Y=-\mathrm{R i c}(Y,\cdot)+\frac{1}{2\tau}Y \quad (7.8) \]