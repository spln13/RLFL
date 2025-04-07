import torch
from server.server import Server
from client.client import Client


def main():
    # 参数初始化
    model = 'MiniVGG'
    save_path = 'model'
    batch_size = 64
    lr = 0.0001
    gamma = 0.99
    k_epochs = 5
    eps_clip = 0.2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    client_num = 10
    pr_list = [0.3, 0.5, 0.7]

    clients = []
    for i in range(client_num):
        client = Client(i, device, model, 1, 1, save_path, batch_size, lr, gamma, k_epochs, eps_clip)
        clients.append(client)

    server = Server(device, clients, 'cifar10', 10, pr_list)
    server.run()

main()