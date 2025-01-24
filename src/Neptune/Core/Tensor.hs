{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}

module Neptune.Core.Tensor where

import Jaxpr (BlxTensor)
import Jaxpr.Blx.Tensor qualified as Blx

class TensorType a where
    symbol :: a -> String

-- a tensor is a vector
class (TensorType b) => BehavesLikeATensor a b where
    -- shape :: a -> [Int]
    -- promoteToShape :: a -> [Int] -> a
    -- nameOf :: a -> String
    tensorType :: a -> b
    castTo :: a -> b -> a

-- renameTensor :: a -> String -> a
-- addTensor :: a -> a -> a
-- subTensor :: a -> a -> a
-- elementWiseMultiply :: a -> a -> a

-- rankOf :: (BehavesLikeATensor a b) => a b -> Int
-- rankOf = length . shape

-- tensorCopyWithNewName :: (BehavesLikeATensor a b) => a b -> String -> a b
-- tensorCopyWithNewName = renameTensor

-- showTensor :: (BehavesLikeATensor a b) => a b -> String
-- showTensor t = nameOf t ++ "[" ++ show (shape t) ++ "]"

instance TensorType Blx.TensorType where
    symbol = show

data StringType = Str1 | Str2 deriving (Show)

data NumType = Num1 | Num2 deriving (Show)

instance TensorType StringType where symbol = show

-- instance TensorType NumType where symbol = show

data FakeTensor = FakeTensor String StringType deriving (Show)

instance BehavesLikeATensor FakeTensor StringType where
    castTo (FakeTensor a _) = FakeTensor a
    tensorType (FakeTensor _ t) = t

-- instance BehavesLikeATensor Blx.BlxTensor Blx.TensorType where
