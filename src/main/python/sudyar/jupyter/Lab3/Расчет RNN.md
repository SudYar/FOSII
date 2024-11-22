---
created: 2024-11-16T19:30
updated: 2024-11-23T00:45
---

## Пример расчета 
Пример с $t\in(0;3]$; Слоев RNN - 2. На вход X размерности (3,), Первый слой (3, 2), второй слой (2,1)
## Forward
$h_t=\tanh{([x_t; h_{t-1}]\times W+b)}$

### Пример расчета
$h^0_{1,2}=(0,0,0)$

$h^1_1=f(W_1\times x^1+V_1\times(0)+b_1)=f(z^1_1)=y^1_1$

$h^1_2=f(W_2\times{y^1_1}+V_2\times(0)+b_2)=f(z^1_2)=y^1$

---
$h^2_1=f(W_1\times x^2 +V_1\times h^1_1 + b_1)=f(z^2_1)=y^2_1$

$h^2_2=f(W_2\times{y^2_1}+V_2\times h^1_2+b_2)=f(z^2_2)=y^2$

---
$h^3_1=f(W_1\times x^3 +V_1\times h^2_1 + b_1)=f(z^3_1)=y^3_1$

$h^3_2=f(W_2\times{y^3_1}+V_2\times h^2_2+b_2)=f(z^3_2)=y^3$

---
## Backprop
$\delta_{h_t} =$ error с следующего слоя или loss + $\delta_{h_{t+1}}$

$dW = \sum\limits_{t=i}^{n} {y_t^T\times \delta_{h_t}}$

$d\_b = \sum\delta_{h_t}$

$\delta_{[x_t, h_{t-1}]}=\delta_{h_t}\times W^T$

Дальше делим на $x_t=[:in\_size]$, $h_{t-1}=[in\_size:]$
### Пример расчета
$e^3 = mse(y^3, \widehat{y^3})$

$\delta^3_2=f'(z^3_2)*e^3$

$dW_2^3=y^3_1 \times \delta^3_2$

$dV^3_2=y^2_2\times \delta^3_2$

$\delta^3_1=\delta^3_2 \times W_2 \;* f'(z^3_1)$

$dW_1^3=x^3 \times \delta^3_1$

$dV^3_1=y^2_1\times \delta^3_1$

---
$e^2 = mse(y^2, \widehat{y^2})$

$\delta^2_2=f'(z^2_2)*(e^2 + V_2\times \delta^3_2)$

$dW_2^2=y^2_1 \times \delta^2_2$

$dV^2_2=y^1_2\times \delta^2_2$

$\delta^2_1=(\delta^2_2 \times W_2 + V_1\times\delta^3_1) \;* f'(z^2_1)$

$dW_1^2=x^2 \times \delta^2_1$

$dV^2_1=y^1_1\times \delta^2_1$

---
$e^1 = mse(y^1, \widehat{y^1})$

$\delta^1_2=f'(z^1_2)*(e^1 + V_2\times \delta^2_2)$

$dW_2^1=y^1_1 \times \delta^1_2$

$\delta^1_1=(\delta^1_2 \times W_2 + V_1\times\delta^2_1) \;* f'(z^1_1)$

$dW_1^1=x^1 \times \delta^1_1$

---
$W_2=-\eta \cdot (dW_2^1+dW_2^2+dW_2^3)$
$W_1=-\eta \cdot (dW_1^1+dW_1^2+dW_1^3)$
c
$V_2=-\eta \cdot (dV_2^2+dV_2^3)$
$V_1=-\eta \cdot (dV_1^2+dV_1^3)$


### Backprop in code
$e^3 = mse(y^3, \widehat{y^3})$

$\delta^3_2=f'(z^3_2)*e^3$

$dW_2^3=y^3_1 \times \delta^3_2$

$dV^3_2=y^2_2\times \delta^3_2$

---
$e^2 = mse(y^2, \widehat{y^2})$

$\delta^2_2=f'(z^2_2)*(e^2 + V_2\times \delta^3_2)$

$dW_2^2=y^2_1 \times \delta^2_2$

$dV^2_2=y^1_2\times \delta^2_2$

---
$e^1 = mse(y^1, \widehat{y^1})$

$\delta^1_2=f'(z^1_2)*(e^1 + V_2\times \delta^2_2)$

$dW_2^1=y^1_1 \times \delta^1_2$



