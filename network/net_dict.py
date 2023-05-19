from network.unet2d5 import U_Net2D5
from network.vnet import V_Net
from network.network import U_Net, R2U_Net, AttU_Net, R2AttU_Net

NetDict = {
    "U_Net": U_Net, 
    "R2U_Net": R2U_Net,
    "AttU_Net": AttU_Net, 
    "R2AttU_Net": R2AttU_Net
}

def get_network(network_type, input_channels, output_channels):
    if network_type == "U_Net2D5":
        network = U_Net2D5(input_channels=input_channels, num_classes=output_channels)
    elif network_type == "V_Net":
        network = V_Net(n_channels=input_channels, n_classes=output_channels)
    elif network_type == "U_Net" or network_type == "R2U_Net" or network_type == "AttU_Net" or network_type == "R2AttU_Net":
        network = NetDict[network_type](img_ch=input_channels, output_ch=output_channels)
    else:
        raise RuntimeError
    return network

if "__name__" == "__main__":
    pass