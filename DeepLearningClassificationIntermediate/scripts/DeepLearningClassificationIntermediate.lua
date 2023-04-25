--[[----------------------------------------------------------------------------

  Application Name:
  DeepLearningClassificationIntermediate
                                                                                             
  Summary:
  Classifying images using a deep neural network trained in dStudio[1].
  This sample app contains a trained network and sample images of the dataset
  'SolderJoint' as described in the Support Portal tutorial [2].
   
  How to Run:
  Starting this sample is possible either by running the app (F5) or
  debugging (F7+F10). Setting breakpoint on the first row inside the 'main'
  function allows debugging step-by-step after 'Engine.OnStarted' event.
  
  More Information:
  [1] https://dstudio.cloud.sick.com/
  [2] Tutorial "dStudio - A step by step example".

------------------------------------------------------------------------------]]
local viewer = View.create()

local colors =
{
  BadContact = {255, 20, 20},
  GoodContact= {20, 255, 20},
  NoContactClean = {20, 255, 255},
  NoContactDirty = {255, 255, 20},
  NoPad = {255, 20, 20},
  ShortCircuit = {255, 20, 20}
}

---@param patch Shape
---@param label string
local function drawClassifiedPatch(patch, label)
  -- Select color based on class label
  ---@type int
  local r, g, b = table.unpack(colors[label])
  local patchDecorator = View.ShapeDecoration.create():setFillColor(r, g, b, 60):setLineColor(r, g, b, 255)
  viewer:addShape(patch, patchDecorator)

  -- Calculate placement relative rectangle and put label in viewer
  local fontSize = 45
  local center, _, height, rotation = Shape.getRectangleParameters(patch)
  local boxBottom = center:add(Point.create(-fontSize/3, height/2 + 10))
  local transform = Transform.createRigid2D(rotation, 0, 0, center)
  local textPosition = boxBottom:transform(transform)
  local textDecoration = View.TextDecoration.create():setSize(fontSize):setColor(255,255,255)
  textDecoration:setPosition(textPosition:getX(), textPosition:getY()):setRotation(-math.pi/2 - rotation)
  textDecoration:setHorizontalAlignment("LEFT")
  viewer:addText(label, textDecoration)
end

---@param image Image
---@param contours Shape[]
---@param patches? Shape[]
local function drawImageWithOverlays(image, contours, patches)
  ---@param shapes Shape[]
  ---@param r int
  ---@param g int
  ---@param b int
  local function drawShapes(shapes, r, g, b)
    local patchDecorator = View.ShapeDecoration.create():setFillColor(r, g, b, 60):setLineColor(r, g, b, 255)
    viewer:addShape(shapes, patchDecorator)
  end

  -- Display the teach configuration
  local locatorDecorator = View.ShapeDecoration.create():setLineColor(20, 20, 255, 255):setLineWidth(5)

  viewer:addImage(image)
  viewer:addShape(contours, locatorDecorator)
  if patches then
    drawShapes(patches, 20, 20, 255)
  end
  viewer:present("ASSURED")
end

---@param teachImage Image
---@return Image.Matching.EdgeMatcher
---@return Image.Fixture
local function teachPart(teachImage)
  -- Construct the input patches for classification in
  -- the coordinate system of the teach image.
  local center = Point.create(554, 570) -- Center point of the first solder joint
  local deltaX = 90                     -- The distance between each joint in pixels
  local height = 1.8*deltaX             -- The height of the inspected areas
  local width = deltaX + 2*5            -- The width of the inspected areas
  local jointCount = 12                 -- Number of joint regions to place
  ---@type Shape[]
  local patches = {}
  for i = 1, jointCount do
    local rectangle = Shape.createRectangle(center, width, height)
    center:setX(center:getX() + deltaX)
    patches[i] = rectangle
  end

  -- Teach a matcher
  local leftRegion = Image.PixelRegion.createRectangle(15, 115, 445, 800)
  local rightRegion = Image.PixelRegion.createRectangle(925, 0, 1422, 475)
  local teachRegion = leftRegion:getUnion(rightRegion)
  local matcher = Image.Matching.EdgeMatcher.create()
  matcher:setMaxMatches(1)
  matcher:setEdgeThreshold(50)
  local teachPose = matcher:teach(teachImage, teachRegion)
  ---@type Shape[]
  local teachContours = matcher:getModelContours():transform(teachPose)

  -- Attach a fixture to handle the transformation of the patch regions
  local fixture = Image.Fixture.create()
  fixture:setReferencePose(teachPose)
  for i, patch in ipairs(patches) do
    fixture:appendShape("patch_" .. i, patch)
  end

  -- Display the teach configuration
  drawImageWithOverlays(teachImage, teachContours, patches)

  return matcher, fixture
end

---@param matcher Image.Matching.EdgeMatcher
---@param fixture Image.Fixture
---@param liveImage Image
---@return Shape[]
local function locatePart(matcher, fixture, liveImage)
  -- Locate part in the new image using the matcher
  local matchPoses = matcher:match(liveImage)

  -- Update the fixture so that objects can be retrieved in the new coordinate frame
  fixture:transform(matchPoses[1])

  -- Retrieve all patches
  local patches = {}
  for i = 1, 12 do
    patches[i] = fixture:getShape("patch_" .. i)
  end

  -- Display some feedback on the localized part
  ---@type Shape[]
  local liveContours = matcher:getModelContours():transform(matchPoses[1])
  drawImageWithOverlays(liveImage, liveContours)
  return patches
end

---@param dnn MachineLearning.DeepNeuralNetwork
---@param image Image
---@param patch Shape
---@return string
local function runInference(dnn, image, patch)
  local patchImage = image:cropRectify(patch) -- Extract the input patch
  dnn:setInputImage(patchImage)               -- Prepare image for prediction
  local result = dnn:predict()                -- Run prediction using the trained network

  -- Parse the result as a classification output
  local predicted_class_idx, predicted_score, predicted_label = result:getAsClassification()
  print(
      string.format(
          "Classified as class number %i: %s with score %0.2f",
          predicted_class_idx,
          predicted_label,
          predicted_score
      )
  )
  return predicted_label
end

local function main()
  -- Load a network from resources
  local net = Object.load("resources/SolderJoint_check_appspace.json") -- This will be a MachineLearning.DeepNeuralNetwork.Model instance
  -- Create an inference engine instance
  local dnn = MachineLearning.DeepNeuralNetwork.create()
  dnn:setModel(net) -- Load the neural network into the engine

  -- Use one image for teaching the locator
  local teachImage = Image.load("resources/images/teach.png")
  local matcher, fixture = teachPart(teachImage)

  -- Rest for 2 seconds
  Script.sleep(2000)

  -- ... Load/Create the rest of your images ...
  local imageDir = "resources/images/SolderJoint" -- Directory path of images
  local imagePaths = File.listRecursive(imageDir) -- List of images

  for _, imagePath in ipairs(imagePaths) do -- For each image
    local image = Image.load(imageDir .. '/' .. imagePath) -- Load image

    -- Get all regions to classify
    local regions = locatePart(matcher, fixture, image)
    for _, region in ipairs(regions) do
      local label = runInference(dnn, image, region)
      drawClassifiedPatch(region, label)
    end

    viewer:present("ASSURED")
    Script.sleep(2000)
  end

end
Script.register("Engine.OnStarted", main)
-- serve API in global scope
