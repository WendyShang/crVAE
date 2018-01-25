require 'torch'
require 'image'

local img = {}


function img.HorizontalFlip(prob)
   return function(input)
      if torch.uniform() < prob then
         input = image.hflip(input)
      end
      return input
   end
end

return img
