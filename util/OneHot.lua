local OneHot, parent = torch.class('OneHot', 'nn.Module')
-- adapted from https://github.com/karpathy/char-rnn/blob/master/util/OneHot.lua

function OneHot:__init(outputSize)
  parent.__init(self)
  self.outputSize = outputSize
  -- We'll construct one-hot encodings by using the index method to
  -- reshuffle the rows of an identity matrix. To avoid recreating
  -- it every iteration we'll cache it.
  self._eye = torch.zeros(outputSize+1,outputSize)
  self._eye[{{1,outputSize},{1,outputSize}}] = torch.eye(outputSize)
  self._eye[outputSize+1] = torch.zeros(outputSize)
end

function OneHot:updateOutput(input)
  self.output:resize(input:size(1), input:size(2), self.outputSize):zero()
  for i = 1,input:size(1) do
      self._eye = self._eye:float()
      local longInput = input[i]:long()
      self.output[i]:copy(self._eye:index(1, longInput))
  end
  return self.output
end
