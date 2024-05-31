import torch
import torch.nn as nn
import torchvision
from matplotlib import pyplot as plt
import torch.nn.functional as F



#liczba epok
n_epochs = 1
#rozmiar partii danych
batch_size = 64
#rozmiar partii danych używany do testowania modelu
batch_size_test = 1
#szybkość aktualizacji parametrów modelu
learning_rate = 0.001
#co ile kroków model wyświetli logi
log_interval = 10
#jak sama nazwa wskazuje random seed
random_seed = 123

out_channels3 = [5,10,25,50,100]
out_channels2 = [20,30,50,100,150]

device = torch.device("cuda" if torch.cuda.is_available() else torch.device('cpu'))
num_of_devices = torch.cuda.device_count()
print(f"Number of devices is {num_of_devices}")
#print(f"Device name is {torch.cuda.get_device_name()}")

# Ładowanie cyfry MINST

#odchylenie standardowe
std = (0.3081)

#wartość średnia
mu = (0.1307,)

#używam dataloader
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('/filse/', train=True, download=True, transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mu,std)
    ])),
    batch_size = batch_size, shuffle = True
)

#test loader
test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('/filse/', train=False, download=True, transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mu,std)
    ])),
    batch_size = batch_size, shuffle = True
)

#sprawdzam dostęp do danych
data, target = next(iter(train_loader))
print(f"Data shape is {data.shape}")
print(f"Shape of target is {target.shape}")
plt.imshow(next(iter(train_loader))[0][6,0,:,:], cmap='gray')
plt.show()

#implementacja klasyfikatora
class Network(nn.Module):
    def __init__(self, out_channels3, out_channels2):
        super(Network, self).__init__()
        #warstwa konwulacyjna
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=40, kernel_size=9)
        self.conv2 = nn.Conv2d(in_channels=40, out_channels=out_channels2, kernel_size=9)

        #regularyzacja, domyślnie wyłącza 50% neuronów w każdej warstwie
        self.conv2_drop = nn.Dropout2d()

        #obliczenie rozmiaru tensora wejściowego dla warstwy liniowej linear1
        self._calculate_linear_input_size()

        #warstwy liniowe
        self.linear1 = nn.Linear(self.linear1_input_size, out_channels3)
        self.linear2 = nn.Linear(out_channels3, 10)

    def _calculate_linear_input_size(self):
        output_size = ((28 - 8) // 2 - 8) // 2  # Obliczenie rozmiaru tensora po dwóch operacjach max-pooling
        self.linear1_input_size = self.conv2.out_channels * output_size * output_size

    # funkcja przetwarzania danych w przód
    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = F.dropout(x, p=0.5, training=self.training)
        x = x.view(-1, self.linear1_input_size)
        x = self.linear1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.linear2(x)
        return F.log_softmax(x, dim=1)


a = next(iter(train_loader))
print(a[0].shape)
print(type(a[0]))


#do wizualizacji postępów uczenia
train_loss = []
train_counter = []
test_loss = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]


#trening
def training_single_epoch(epoch, model, optimizer, train_loader, loss_fcn):
    #ustawiam model w tym do uczenia jest to niezbędne aby dropout działał
    model.train()
    for batch_ind, (data, target) in enumerate(train_loader):
        #zeruje gradienty z poprzedniej iteracji
        optimizer.zero_grad()

        # przetwarzanie do przodu, przekazuje dane przez model, generując przewidywania
        output = model(data)
        #oblicza stratę między przewidywanymi a rzeczywistymi wartościami.
        loss = F.nll_loss(output, target)

        # propagacja do tyłu, oblicza gradienty błędu względem parametrów modelu
        loss.backward()

        # uaktualnienie parametrów
        optimizer.step()

        #jeżeli index batcha jest wielokrotnością log_interval to wyświetla informacje o bieżącej epoce, postępie w danych treningowych oraz wartości straty.
        if batch_ind % log_interval == 0:
            print("Train Epoch: {}[{}/{} ({:.0f}%)]\tLoss: {:.6f}]".format(
                epoch, batch_ind * len(data), len(train_loader.dataset),
                100. * batch_ind / len(train_loader), loss.item()
            ))


# testowanie
def test():
    #ustawiam model w tryb testowania bo inaczej dropout nie zadziała
    model.eval()
    #inicjalizacja zmiennych do zapisu całkowitej straty testowej i liczby poprawnych przewidywań
    test_loss = 0
    correct = 0
    test_losses = []

    with torch.no_grad():
        #iteracja przez wszystkie serie danych z test_loader
        for data, target in test_loader:
            #przetwarzanie danych w przód, przekazuje dane przez model generujac przewidywania
            output = model(data)
            #dodawanie wartości staraty z obecnej serii do całkowitej straty testoej
            test_loss += F.nll_loss(output, target).item()

            #obliczam poprawne przewidywania olbiczając liczbę poprawnych przewidywań w bieżącej serii
            pred = output.data.max(dim = 1, keepdim = True)[1]

            # suma poprawnych predykcji czyli przewidywań
            correct += pred.eq(target.data.view_as(pred)).sum()

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    print("\nTest set: Acg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)
    ))

    # obliczanie dokładności dla całego zbioru testowego
    accuracy = 100. * correct / len(test_loader.dataset)
    # zwracanie dokładności i straty testowej
    return test_loss, accuracy

# trenujemy nasz model


results = []


# Testowanie modelu dla każdej kombinacji parametrów
#for channels1, channels2 in zip(out_channels1, out_channels2):
# Testowanie modelu dla każdej kombinacji parametrów
for channels3 in out_channels3:
    for channels2 in out_channels2:
        model = Network(channels3, channels2)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

        print(f"\nTesting model with out_channels1={channels3} and out_channels2={channels2}")
        test_accuracy = []
        for epoch in range(1, n_epochs + 1):
            training_single_epoch(epoch, model=model, optimizer=optimizer, train_loader=train_loader, loss_fcn=F.nll_loss)
            test_loss, accuracy = test()  # pobranie dokładności i straty testowej
            test_accuracy.append(accuracy)  # zapisanie dokładności w liście
            print(f"Test Loss: {test_loss}, Accuracy: {accuracy}%")
        results.append((channels3, channels2, test_accuracy))

data, label = next(iter(test_loader))
with torch.no_grad():
    output = model(data)
    fig = plt.figure()
    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.tight_layout()
        plt.imshow(torch.squeeze(data[i][0]), cmap = 'gray', interpolation = 'none')
        plt.title("prediction: {}".format(
            output.data.max(1, keepdim = True)[1][i].item()
        ))
        plt.xticks([])
        plt.yticks([])
        plt.show()


# Tworzenie wykresu 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Pobranie wyników treningu
for result in results:
    channels3, channels2, accuracy = result
    # Dodanie punktów do wykresu
    ax.plot([channels3], [channels2], [accuracy[-1]], marker='o', color='r')

# Ustawienie etykiet osi
ax.set_xlabel('out_channels3')
ax.set_ylabel('out_channels2')
ax.set_zlabel('Accuracy')

# Zmiana punktów na wykresie na powierzchnię
X = [result[0] for result in results]
Y = [result[1] for result in results]
Z = [result[2][-1] for result in results]
ax.plot_trisurf(X, Y, Z, cmap='viridis', edgecolor='none')

plt.show()

