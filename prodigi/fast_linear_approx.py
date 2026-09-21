import numpy as np
import torch

class FastLinearApprox():
    def __init__(self, func, minv=0, maxv=1, num_points=10000):
        self.func = func
        self.minv = minv
        self.maxv = maxv
        self.num_points = num_points
        self.x_values = np.linspace(minv, maxv+1e-8, num_points)
        self.y_values = func(self.x_values)
        #self.slopes = np.gradient(self.y_values, self.x_values)
        self.delta=self.x_values[1] - self.x_values[0]

    def __call__(self, values):
        #finds the closest two indices (below and above) in the precomputed x_values for each value in values, and uses them to approximate the value of the function/y at that point using linear interpolation
        values = np.asarray(values)
        # Clip values to be within the range of x_values
        clipped_values = np.clip(values, self.minv, self.maxv)
        # Find the indices of the closest points in x_values
        # you know that x is linspace, so we dont need search_sorted
        indices = ((clipped_values - self.minv) / (self.maxv - self.minv) * (self.num_points - 1))
        indices_above=np.ceil(indices).astype(int)
        indices_below=np.floor(indices).astype(int)
        # Get the x and y values for the closest points
        x_below = self.x_values[indices_below]
        #x_above = self.x_values[indices_above]
        y_below = self.y_values[indices_below]
        y_above = self.y_values[indices_above]
        # Perform linear interpolation
        t= (clipped_values - x_below) / self.delta
        approx_values = y_below + t * (y_above - y_below)
        return approx_values


class FastLinearApprox2d():
    def __init__(self, func, minv=0, maxv=1, minw=0, maxw=1, num_points=1000):
        self.func = func
        self.minv = minv
        self.maxv = maxv
        self.minw = minw
        self.maxw = maxw
        self.num_points = num_points
        self.xv_values = np.linspace(minv, maxv+1e-8, num_points)
        self.xw_values = np.linspace(minw, maxw+1e-8, num_points)
        #I want a array containing all combination sof xv and xw values. Each row contains [xv, xw] for a specific combination. So I can use this to evaluate the function at all combinations of xv and xw. I can use np.meshgrid to do this.
        self.x_values=np.array(np.meshgrid(self.xv_values, self.xw_values)).T.reshape(-1, 2)
        self.y_values = func(self.x_values[:, 0], self.x_values[:, 1]).reshape(num_points, num_points)
        #self.slopes = np.gradient(self.y_values, self.x_values)
        self.deltav=self.xv_values[1] - self.xv_values[0]
        self.deltaw=self.xw_values[1] - self.xw_values[0]

    def __call__(self, values1, values2):
        #finds the closest two indices (below and above) in the precomputed x_values for each value in values, and uses them to approximate the value of the function/y at that point using linear interpolation
        values1 = np.asarray(values1)
        values2 = np.asarray(values2)
        # Clip values to be within the range of x_values
        clipped_values1 = np.clip(values1, self.minv, self.maxv)
        clipped_values2 = np.clip(values2, self.minw, self.maxw)
        #since this 2d, we need to find 4 points in total: above in v, below in v, above in w, below in w
        indices_v = ((clipped_values1 - self.minv) / (self.maxv - self.minv) * (self.num_points - 1))
        indices_w = ((clipped_values2 - self.minw) / (self.maxw - self.minw) * (self.num_points - 1))
        indices_v_above=np.ceil(indices_v).astype(int)
        indices_v_below=np.floor(indices_v).astype(int)
        indices_w_above=np.ceil(indices_w).astype(int)
        indices_w_below=np.floor(indices_w).astype(int)
        # Get the x and y values for the closest points
        xv_below = self.xv_values[indices_v_below]
        #xv_above = self.xv_values[indices_v_above]
        xw_below = self.xw_values[indices_w_below]
        #xw_above = self.xw_values[indices_w_above]
        y_below_below = self.y_values[indices_v_below, indices_w_below]
        y_below_above = self.y_values[indices_v_below, indices_w_above]
        y_above_below = self.y_values[indices_v_above, indices_w_below]
        y_above_above = self.y_values[indices_v_above, indices_w_above]
        t_v= (clipped_values1 - xv_below) / self.deltav
        t_w= (clipped_values2 - xw_below) / self.deltaw
        # Perform bilinear interpolation
        approx_values = (1 - t_v) * (1 - t_w) * y_below_below + \
                        (1 - t_v) * t_w * y_below_above + \
                        t_v * (1 - t_w) * y_above_below + \
                        t_v * t_w * y_above_above
        return approx_values

