from torch import nn


class SimpleCNN(nn.Module):
    def __init__(self, input_channels: int = 1, output_channels: int = 64,
                 kernel_size: int = 3, n_hidden: int = 128, dropout: float = 0.5):

        """
        Initialize the SimpleCNN model.

        Parameters:
            input_channels (int): number of input channels
            output_channels (int): number of output channels
            kernel_size (int): size of the convolutional kernel
            n_hidden (int): number of hidden units
            dropout (float): dropout rate

        """
        super(SimpleCNN, self).__init__()

        # Define the layers of the backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size),
            nn.LeakyReLU(),
            nn.Conv2d(output_channels, output_channels//2, kernel_size),
            nn.LeakyReLU(),
            nn.Conv2d(output_channels//2, output_channels//4, kernel_size),
            nn.LeakyReLU(),
            nn.Conv2d(output_channels//4, output_channels//8, kernel_size),
            nn.LeakyReLU()
        )

        # Define the classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(output_channels//8 * 20 * 20, n_hidden),
            nn.Dropout(dropout),
            nn.Linear(n_hidden, 10)
        )

    def forward(self, x):
        """
        Define the forward pass of the model.

        Parameters:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output tensor
        """
        return self.classifier(self.backbone(x))
