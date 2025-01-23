module Jaxpr.Blx.Tensor where

import Data.List (intercalate)

data TensorType = Tf32 | Tf16 deriving (Eq)

instance Show TensorType where
    show Tf32 = "f32"
    show Tf16 = "f16"

type Axis = Int -- in future might be literal

type AxisSize = Int -- in future might be literal

type Shape = [AxisSize]

showAxisWithParenthesis :: [Int] -> String
showAxisWithParenthesis as = "(" ++ intercalate "," (map show as) ++ ")"

data Designation = Tvar | Tlit

data BlxTensor = BlxTensor TensorType [AxisSize] String Designation

instance Show BlxTensor where
    show (BlxTensor t s n _) = n ++ ":" ++ show t ++ show s

tensor :: TensorType -> [Int] -> String -> Designation -> BlxTensor
tensor = BlxTensor

tensorRank :: BlxTensor -> Int
tensorRank = length . tensorShape

tensorShape :: BlxTensor -> [AxisSize]
tensorShape (BlxTensor _ s _ _) = s

tensorType :: BlxTensor -> TensorType
tensorType (BlxTensor t _ _ _) = t

sameType :: BlxTensor -> BlxTensor -> Bool
sameType t1 t2 = tensorType t1 == tensorType t2

sameShape :: BlxTensor -> BlxTensor -> Bool
sameShape t1 t2 = tensorShape t1 == tensorShape t2

shapeAtAxis :: BlxTensor -> Axis -> AxisSize
shapeAtAxis t n = tensorShape t !! n

tensorName :: BlxTensor -> String -- identifier
tensorName (BlxTensor _ _ n _) = n

renameTensor :: BlxTensor -> String -> BlxTensor
renameTensor (BlxTensor t s _ z) newName = BlxTensor t s newName z

tensorCopy :: BlxTensor -> BlxTensor
tensorCopy t = renameTensor t (tensorName t ++ ".copy")

renameTensorsWithSeedName :: [BlxTensor] -> String -> [BlxTensor]
renameTensorsWithSeedName ts seedName = renamedTensors
  where
    newNames = map (((seedName ++ ".") ++) . show) [1, 2 .. (length ts)]
    renamedTensors = zipWith renameTensor ts newNames

tensorDesignation :: BlxTensor -> Designation
tensorDesignation (BlxTensor _ _ _ vt) = vt

shapeSingleConcat :: Shape -> Shape -> Axis -> Shape
shapeSingleConcat (a : as) (b : bs) axis = f (a : as) (b : bs) axis 0
  where
    f :: Shape -> Shape -> Axis -> Axis -> Shape
    f (c : cs) (d : ds) target cur
        | target == cur =
            if cs == ds
                then c + d : cs
                else error "shape mismatch"
        | target > cur =
            if c == d
                then c : f cs ds target (cur + 1)
                else error "shape mismatch"
        | otherwise = error "Not enough axes"
    f _ _ _ _ = error "Should not happen"
shapeSingleConcat _ _ _ = error "Shape should be non empty list"

shapeConcat :: [Shape] -> Axis -> Shape
shapeConcat (shap : shapes) axis = foldl (\x y -> shapeSingleConcat x y axis) shap shapes
shapeConcat [] _ = error "no shapes to concat"
