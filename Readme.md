# Statespace

This is essentially a Python port of [MARSS](https://github.com/atsa-es/MARSS/), though it isn't aiming for exact feature/interface parity. There is a fair amount of model validation code in MARSS that is not yet present in this library, so you need to be a lot more careful about model specification if you're going to use this. This library will let you do a lot of things that aren't valid, and mostly that will result in your fit diverging pretty obviously, but not necessarily always. I hope to add better validation over time.

You can find some detail about which models are and are not admissible [here](https://cran.r-project.org/web/packages/MARSS/vignettes/EMDerivation.pdf).

## Example

This is a simple ARMA(1, 1) model, represented in the [Hamilton form](http://www-stat.wharton.upenn.edu/~stine/stat910/lectures/14_state_space.pdf)
```python
y = arma_generate_sample(ar=[1, -0.4], ma=[1, 0.2], nsample=1000, scale=1)

kf = KalmanFilter(
	y, 
	B = np.array([
		['phi', 0],
		[1.0, 0]
	]),
	Q = np.array([
		[1.0, 0],
		[0, 0]
	]),
	Z = np.array([[1.0, 'theta']]),
	R = np.array([['measurement_noise']])
)

kf.fit()
```

This should output something like:

```
Starting EM on: B, Z, R
	[0] ll: -1867.349491
	[10] ll: -1462.598377
	[20] ll: -1438.460797
	[30] ll: -1428.766103
	[40] ll: -1423.972850
	[50] ll: -1421.256981
	[60] ll: -1419.564740
	[70] ll: -1418.434631
	[80] ll: -1417.639221
	[90] ll: -1417.055898
	[100] ll: -1416.613761
	[110] ll: -1416.269444
	[120] ll: -1415.995182
	[130] ll: -1415.772502
	[140] ll: -1415.588714
	[150] ll: -1415.434856
	[160] ll: -1415.304445
	[170] ll: -1415.192692
	[180] ll: -1415.095996
Fitted:
	phi: 0.4616
	theta: 0.2144
	measurement_noise: 0.0427

Minimizing: B, Z, R
Starting likelihood: -1415.10 (-1415.10)
Minimized: -1413.78
Fitted:
	phi: 0.4822
	theta: 0.1567
	measurement_noise: 0.0000
```

`.fit()` uses a combination of the EM algorithm and BFGS to automatically fit your model. It is just a convenience wraper for:

```python
kf.em(tol=0.1)
kf.minimize()
````