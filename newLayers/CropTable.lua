local CropTable, parent = torch.class('nn.CropTable', 'nn.Module')

function CropTable:__init(dimension, nChunk ,nInputDims)
   parent.__init(self)
   self.dimension = dimension
   self.nInputDims = nInputDims
   self.nChunk = nChunk
end

function CropTable:_getPositiveDimension(input)
   local dimension = self.dimension
   if dimension < 0 then
      dimension = input:dim() + dimension + 1
   elseif self.nInputDims and input:dim()==(self.nInputDims+1) then
      dimension = dimension + 1
   end
   return dimension
end

function CropTable:updateOutput(input)
   local dimension = self:_getPositiveDimension(input)
   local slices = input:size(dimension)
   local chunk_border = math.floor(slices/self.nChunk);
   assert(chunk_border~=0, "The input size is less than the number of chunk")

   local currentOutput= {}
   local crop_index = {}
   for i=1,slices do
      crop_index[#crop_index+1] = i;
      if (i % chunk_border == 0) or (i ==sclices) then 
         currentOutput[#currentOutput+1] = input:index(dimension,torch.LongTensor(crop_index))
         crop_index ={};
      end
   end
   self.output = currentOutput
   return self.output
end 

function CropTable:updateGradInput(input, gradOutput)
   local dimension = self:_getPositiveDimension(input)
   local slices = input:size(dimension)
   local chunk_border = math.floor(slices / self.nChunk);
   self.gradInput:resizeAs(input)
   local crop_index = {}
   local chunk = 0 ;
   for i=1,slices do
      crop_index[#crop_index+1] = i; 
      if (i % chunk_border == 0) or (i ==sclices) then 
         chunk = chunk +1;
         local currentGradInput = gradOutput[chunk];              
         self.gradInput:indexCopy(dimension,torch.LongTensor(crop_index),currentGradInput)
         crop_index = {}
      end
   end
   return self.gradInput
end