#!/usr/bin/env fish

set echoer "import hiddenlayer as hl
from torch.autograd import Variable
x = Variable(torch.rand(1, 1, 28, 28))
n = Net()
n.eval()
h = hl.build_graph(n, x)
h.save(gp.png)"

 for i in (ls)
    for j in (ls $i)
             if [ "$j" = "Nets.py" ]
            # echo $echoer >> $i/$j
            cat $i/$j/ gp.png/'gp.png'
        end
        end
        # break
end
