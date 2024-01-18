// To run this code, you need to download an appropriate libtorch
// https://github.com/dotnet/TorchSharp#download
// This project has 'libtorch-cuda-12.1-linux-x64' installed by default

// For some guidance on how to use the TorchSharp library
// Check out their tutorial/example repository
// https://github.com/dotnet/TorchSharpExamples/tree/main/tutorials/CSharp

using TorchSharpChapter08;
using static TorchSharp.torch;

var X = rand(2, 20);

var net = new MLP("myMlp");

Console.WriteLine($"[{string.Join(", ", net.call(X).shape)}]");


var net2 = new MySequential(
    nn.Linear(20, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
);

X = rand(2, 20);

Console.WriteLine($"[{string.Join(", ", net2.call(X).shape)}]");
