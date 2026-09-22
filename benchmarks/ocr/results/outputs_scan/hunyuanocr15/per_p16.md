where $ H(X) $ is the Hamilton's expression for the trace Harnack inequality (with $ t = - \tau $). Hence,

$$ \bar{\tau}^{\frac{3}{2}}(R + |X|^{2})(\bar{\tau}) = -K + \frac{1}{2}L(q, \bar{\tau}),  \tag{7.4} $$

where $ K = K(\gamma, \bar{\tau}) $ denotes the integral $ \int_{0}^{\bar{\tau}}\tau^{\frac{3}{2}}H(X)d\tau $, which we'll encounter a few times below. Thus we get

$$ L_{\bar{\tau}} = 2\sqrt{\bar{\tau}}R - \frac{1}{2\bar{\tau}}L + \frac{1}{\bar{\tau}}K  \tag{7.5} $$

$$ |\nabla L|^{2} = -4\bar{\tau}R + \frac{2}{\sqrt{\bar{\tau}}}L - \frac{4}{\sqrt{\bar{\tau}}}K  \tag{7.6} $$

Finally we need to estimate the second variation of $ L $. We compute

$$ \delta_{Y}^{2}(\mathcal{L}) = \int_{0}^{\bar{\tau}}\sqrt{\tau}(Y \cdot Y \cdot R + 2 < \nabla_{Y}\nabla_{Y}X, X > + 2|\nabla_{Y}X|^{2})d\tau $$

$$ =\int_{0}^{\bar{\tau}}\sqrt{\tau}(Y\cdot Y\cdot R+2<\nabla_{X}\nabla_{Y}Y,X>+2<R(Y,X),Y,X>+2|\nabla_{X}Y|^{2})d\tau $$

Now

$$ \frac{d}{d\tau}<\nabla_{Y}Y,X>==<\nabla_{X}\nabla_{Y}Y,X>+<\nabla_{Y}Y,\nabla_{X}X>+2Y\cdot\text{Ric}(Y,X)-X\cdot\text{Ric}(Y,Y), $$

so, if $ Y(0) = 0 $ then

$$ \delta_{Y}^{2}(\mathcal{L}) = 2 < \nabla_{Y}Y, X > \sqrt{\bar{\tau}} + $$

$$ \int_{0}^{\bar{\tau}}\sqrt{\tau}(\nabla_{Y}\nabla_{Y}R+2<R(Y,X),Y,X>+2|\nabla_{X}Y|^{2}\\+2\nabla_{X}\text{Ric}(Y,Y)-4\nabla_{Y}\text{Ric}(Y,X))d\tau, \tag{7.7} $$

where we discarded the scalar product of $ -2\nabla_{Y}Y $ with the left hand side of (7.2). Now fix the value of $ Y $ at $ \tau = \bar{\tau} $, assuming $ |Y(\bar{\tau})| = 1 $, and construct $ Y $ on $ [0, \bar{\tau}] $ by solving the ODE

$$ \nabla_{X}Y = -\text{Ric}(Y,\cdot) + \frac{1}{2\tau}Y  \tag{7.8} $$