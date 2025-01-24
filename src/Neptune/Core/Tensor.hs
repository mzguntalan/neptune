module Neptune.Core.Tensor where

class TensorType a where
    symbol :: a -> String

-- a tensor is a vector
class (TensorType b) => BehavesLikeATensor a b where
    shape :: a b -> [Int]
    promoteToShape :: a b -> [Int] -> a b
    nameOf :: a b -> String
    tensorType :: a b -> b
    castTo :: (TensorType c) => a b -> a c
    renameTensor :: a b -> String -> a b

rankOf :: (BehavesLikeATensor a b) => a b -> Int
rankOf = length . shape

tensorCopyWithNewName :: (BehavesLikeATensor a b) => a b -> String -> a b
tensorCopyWithNewName = renameTensor

showTensor :: (BehavesLikeATensor a b) => a b -> String
showTensor t = nameOf t ++ "[" ++ show (shape t) ++ "]"
