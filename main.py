import torch
from server.server import Server
from client.client import Client


def main():
    # 参数初始化
    model = 'MiniVGG'
    save_path = 'model'
    batch_size = 64
    s = 0.0001
    gamma = 0.99
    k_epochs = 5
    eps_clip = 0.2
    dataset = 'cifar10'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    client_num = 10
    pr_list = [0.5]

    clients = []
    for i in range(client_num):
        client = Client(i, device, model, 1, save_path, dataset, pr_list, batch_size)
        clients.append(client)

    server = Server(device, clients, 'cifar10', 10, pr_list)
    server.run()


main()
