import torch
import torch.nn as nn


class INN(nn.Module):
    """
    Invertible Neural Network (INN) class.

    This class defines an Invertible Neural Network used for bidirectional mapping
    between input parameters (material component, screw speed, extrusion temperature)
    and output (volumetric flow rate and latent variables). It can perform forward
    prediction of the flow rate and inverse optimization of process parameters.

    Attributes:
        w1, w2, w3, w4 (nn.Parameter): Weight matrices for the four fully - connected layers.
        b1, b2, b3, b4 (nn.Parameter): Bias vectors for the four fully - connected layers.
        slope (float): Slope for the Leaky ReLU activation function.
    """

    def __init__(self):
        """
        Initialize the INN model.

        This method initializes the weights and biases of the fully - connected layers
        and sets the slope for the Leaky ReLU activation function.
        """
        super(INN, self).__init__()
        self.w1 = nn.Parameter(torch.randn(3, 3))
        self.b1 = nn.Parameter(torch.randn(1, 3))
        self.w2 = nn.Parameter(torch.randn(3, 3))
        self.b2 = nn.Parameter(torch.randn(1, 3))
        self.w3 = nn.Parameter(torch.randn(3, 3))
        self.b3 = nn.Parameter(torch.randn(1, 3))
        self.w4 = nn.Parameter(torch.randn(3, 3))
        self.b4 = nn.Parameter(torch.randn(1, 3))
        self.slope = 0.03

    def leaky_relu(self, x):
        """
        Apply the Leaky ReLU activation function.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying Leaky ReLU.
        """
        y1 = ((x > 0) * x)
        y2 = ((x <= 0) * x * self.slope)
        return y1 + y2

    def inv_leaky_relu(self, x):
        """
        Apply the inverse of the Leaky ReLU activation function.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the inverse of Leaky ReLU.
        """
        y1 = ((x > 0) * x)
        y2 = ((x <= 0) * x * (1. / self.slope))
        return y1 + y2

    def forward(self, x):
        """
        Perform the forward pass of the INN model.

        Args:
            x (torch.Tensor): Input tensor containing material component,
                              screw speed, and extrusion temperature.

        Returns:
            torch.Tensor: Output tensor representing the predicted flow rate
                          (and latent variables in the context of the overall model).
        """
        l1 = self.leaky_relu(torch.matmul(x, self.w1) + self.b1)
        l2 = self.leaky_relu(torch.matmul(l1, self.w2) + self.b2)
        l3 = self.leaky_relu(torch.matmul(l2, self.w3) + self.b3)
        y = torch.matmul(l3, self.w4) + self.b4
        return y

    def inverse(self, x):
        """
        Perform the inverse pass of the INN model.

        Args:
            x (torch.Tensor): Input tensor, usually the target flow rate.

        Returns:
            torch.Tensor: Output tensor representing the optimized process parameters
                          (material component, screw speed, extrusion temperature).
        """
        l3_back = self.inv_leaky_relu(torch.matmul(x - self.b4, torch.inverse(self.w4)))
        l2_back = self.inv_leaky_relu(torch.matmul(l3_back - self.b3, torch.inverse(self.w3)))
        l1_back = self.inv_leaky_relu(torch.matmul(l2_back - self.b2, torch.inverse(self.w2)))
        x_back = torch.matmul(l1_back - self.b1, torch.inverse(self.w1))
        return x_back
