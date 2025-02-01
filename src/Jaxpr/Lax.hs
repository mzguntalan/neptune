module Jaxpr.Lax where

import Neptune.Core.Program
import Neptune.Core.Tensor2 (shapeConcat)

-- Abstract Tensors

data TensorOrigin a = (Show a, Eq a) => FromValue a | FromAbstract | FromComputation

instance Show (TensorOrigin a) where
    show FromAbstract = "FromAbstract"
    show (FromValue a) = "Val:" ++ show a
    show FromComputation = "FromComputation"

instance Eq (TensorOrigin a) where
    FromAbstract == FromAbstract = False
    FromValue a1 == FromValue a2 = a1 == a2
    FromComputation == FromComputation = False
    _ == _ = False

type Shape = [Int]

data TensorType = Tf32 | Ti32 | Tstr deriving (Eq, Show)

data TensorDescription = TensorDescription Shape TensorType (TensorOrigin String) deriving (Eq, Show)

data LaxPrimitive = LaxPrimitive String [(String, String)] TensorDescription deriving (Eq, Show)

type Tensor = Program LaxPrimitive

laxprimitive :: String -> [(String, String)] -> TensorDescription -> PrimitiveInstruction LaxPrimitive
laxprimitive primName primParams primOutputDescription = PrimitiveInstruction (LaxPrimitive primName primParams primOutputDescription)

abstractTensor :: Shape -> TensorType -> Tensor
abstractTensor shape ty = Immediate (laxprimitive "abtract_tensor" [] (TensorDescription shape ty FromAbstract)) []

getShape :: Tensor -> Shape
getShape t = s
  where
    PrimitiveInstruction (LaxPrimitive _ _ (TensorDescription s _ _)) = lastPrimitiveInstructionInProgram t

getType :: Tensor -> TensorType
getType t = tensorType
  where
    PrimitiveInstruction (LaxPrimitive _ _ (TensorDescription _ tensorType _)) = lastPrimitiveInstructionInProgram t

ladd :: Tensor -> Tensor -> Tensor
ladd a b
    | getShape a == getShape b = Immediate (PrimitiveInstruction (LaxPrimitive "add" [] (TensorDescription shape ty FromComputation))) [a, b]
    | otherwise = error "Shape"
  where
    shape = getShape a
    ty = getType a

isNumeric :: Tensor -> Bool
isNumeric a = getType a `elem` [Tf32, Ti32]

labs :: Tensor -> Tensor
labs a
    | isNumeric a = Immediate (PrimitiveInstruction (LaxPrimitive "abs" [] (TensorDescription shape ty FromComputation))) [a]
    | otherwise = error "not a number"
  where
    shape = getShape a
    ty = getType a

lconcatenate :: [Tensor] -> Int -> Tensor
lconcatenate (t : ts) dim = Immediate (PrimitiveInstruction (LaxPrimitive "concatenate" [("dimension", show dim)] (TensorDescription shape ty FromComputation))) tensors
  where
    shape = shapeConcat (map getShape tensors) dim
    tensors = t : ts
    ty = getType t
lconcatenate [] _ = error "no"

testFunction :: Tensor -> Tensor -> Tensor
testFunction a b = lconcatenate [a, b, c] 0
  where
    c = ladd a b `ladd` z
    z = abstractTensor (getShape a) (getType b)
