local HardSign = torch.class('nn.HardSign', 'nn.Module')

function HardSign:updateOutput(input)
	local s = input:size()
   self.inputMeanabs = input:norm(1,2):div(s[2]):repeatTensor(1,s[2],1,1);
   self.output:resizeAs(input):copy(input):sign():cmul(self.inputMeanabs)
   --self.temp = self.temp or input.new()
   --self.temp:resizeAs(input):copy(input):abs():add(1)
   --self.output:resizeAs(input):copy(input):cdiv(self.temp):cmul(self.inputMeanabs)
   return self.output
end

function HardSign:updateGradInput(input, gradOutput)
   --self.tempgrad = self.tempgrad or input.new()
   --self.tempgrad:resizeAs(self.output):copy(input):abs():add(0.01):cmul(self.tempgrad)
   --self.gradInput:resizeAs(input):copy(gradOutput):cdiv(self.tempgrad):mul(0.0001):cmul(self.inputMeanabs)
   self.gradInput:resizeAs(gradOutput):copy(gradOutput):cmul(self.inputMeanabs)
   return self.gradInput 
end
-- function HardSign:updateOutput(input)
--    self.temp = self.temp or input.new()
--    self.temp:resizeAs(input):copy(input):abs():add(1e-4)
--    self.output:resizeAs(input):copy(input):cdiv(self.temp)
--    return self.output
-- end

-- function HardSign:updateGradInput(input, gradOutput)
--    self.tempgrad = self.tempgrad or input.new()
--    self.tempgrad:resizeAs(self.output):copy(input):abs():add(0.1):cmul(self.tempgrad):div(10)
--    self.gradInput:resizeAs(input):copy(gradOutput)--:cdiv(self.tempgrad)
--    --print(self.tempgrad:mean())
--    return self.gradInput
-- end