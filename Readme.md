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

`.fit()` uses a combination of the EM algorithm and BFGS to automatically fit your model. It is just a convenience wraper for:

```python
kf.em(tol=0.1)
kf.minimize()
````