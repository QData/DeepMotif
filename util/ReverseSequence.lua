------------------------------------------------------------------------
-- Adapted from github.com/jcjohnson/torch-rnn/pull/66/commits/5e30c1d54dc9ed1d152e4a55e9c438775f623ef7

--[[ ReverseSequence ]] --
-- Reverses a sequence on a given dimension.
-- Example: Given a tensor of torch.Tensor({{1,2,3,4,5}, {6,7,8,9,10})
-- nn.ReverseSequence(1):forward(tensor) would give: torch.Tensor({{6,7,8,9,10},{1,2,3,4,5}})
------------------------------------------------------------------------
local ReverseSequence, parent = torch.class("nn.ReverseSequence", "nn.Module")

function ReverseSequence:__init(dim,gpu)
    parent.__init(self)
    self.output = torch.Tensor()
    self.gradInput = torch.Tensor()
    self.outputIndices = torch.LongTensor()
    self.gradIndices = torch.LongTensor()
    self.typ = 'torch.CudaTensor'
    if gpu and (gpu < 1) then
      self.typ = 'torch.LongTensor'
    end
end

function ReverseSequence:reverseOutput(input)
  self.output:resizeAs(input)
  self.outputIndices:resize(input:size())
  local T = input:size(1)
  for x = 1, T do
      self.outputIndices:narrow(1, x, 1):fill(T - x + 1)
  end
  self.output:gather(input, 1, self.outputIndices:type(self.typ))
end

function ReverseSequence:updateOutput(input)
  input = input:transpose(1, 2)
  self:reverseOutput(input)
  self.output = self.output:transpose(1, 2)
  return self.output
end

function ReverseSequence:reverseGradOutput(gradOutput)
    self.gradInput:resizeAs(gradOutput)
    self.gradIndices:resize(gradOutput:size())
    local T = gradOutput:size(1)
    for x = 1, T do
        self.gradIndices:narrow(1, x, 1):fill(T - x + 1)
    end
    self.gradInput:gather(gradOutput, 1, self.gradIndices:type(self.typ))
end

function ReverseSequence:updateGradInput(inputTable, gradOutput)
  gradOutput = gradOutput:transpose(1, 2)
  self:reverseGradOutput(gradOutput)
  self.gradInput = self.gradInput:transpose(1, 2)
  return self.gradInput
end
