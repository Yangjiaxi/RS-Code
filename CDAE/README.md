# Collaborative denoising auto-encoders for top-N recommender systems

## Working in progress.

TODO List:

- [x] CDAE model.
- [ ] Load MovieLens-1M with 5x negative-sampling.
- [ ] Training.
- [ ] Exp.

## Info

Year: 2016

Author(s):

- Wu, Yao
- DuBois, Christopher
- Zheng, Alice X.
- Ester, Martin

Journal: WSDM 2016 - Proceedings of the 9th ACM International Conference on Web Search and Data Mining

还是**污染数据+自编码器**模型

自编码器是FC->FC，但是第一步FC除了把某个用户的feedback塞进来，还把该用户专属的一个属性(k维，与中间层节点数量相同，也就是隐变量)当做偏置项加进来

也就是z_u = h(W.T.y_u<污染> + Vu+b)其中Vu是用户u专属的偏置

然后走一个解码函数f

y_ui = f(W'i.T.z_u + b'_i)

对于f()的选择可以体现该模型对其他模型的表征能力

1. f(x) = x，恒等函数，近似LFSM
2. 污染等级q->1，LFM
3. 移除用户节点Vu，FSM
4. liner function，LCR
