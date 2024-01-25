using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace TorchSharpChapter08;

public class MySequential : Module<Tensor, Tensor>
{
    public MySequential(params Module[] args) : base(nameof(MySequential))
    {
        foreach ((Module layer, string i) in args.Select((x, i) => (x, i.ToString())))
        {
            add_module(i.ToString(), layer);
        }
    }

    public override Tensor forward(Tensor input)
    {
        foreach (Module module in children())
        {
            if (module is not Module<Tensor, Tensor> layer)
            {
                throw new InvalidOperationException("Modules must all accept and return tensors");
            }

            input = layer.call(input);
        }

        return input;
    }
}