using static TorchSharp.torch.nn;
using static TorchSharp.torch;
using static TorchSharp.torch.nn.functional;
using TorchSharp.Modules;

namespace TorchSharpChapter08;

public class MLP : Module<Tensor, Tensor>
{
    public MLP(string name) : base(name)
    {
        // LazyLinear is not yet available
        _hidden = Linear(20, 256);
        _output = Linear(256, 10);
    }

    public override Tensor forward(Tensor input) =>
        _output.call(relu(_hidden.call(input)));

    private readonly Linear _hidden;
    private readonly Linear _output;
}
