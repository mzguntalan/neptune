{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}

module Neptune.Core.Tensor where

import Jaxpr (BlxTensor)
import Jaxpr.Blx.Tensor qualified as Blx

class BehaveLikeAType a where
    symbol :: a -> String
    castable :: a -> a -> Bool
    convertible :: a -> a -> Bool
    (==) :: a -> a -> Bool

-- a tensor is a vector
class (BehaveLikeAType b) => Typed a b where
    tensorType :: a -> b
    castTo :: a -> b -> a

class (Num b) => ConcreteTensor a b where -- No idea what this should be yet
    valueOf :: a -> b

class Named a where
    nameOf :: a -> String
    renameTo :: a -> String -> a

class StrictTensor a where -- maybe
    shape :: a -> [Int]

    -- these are probably lax primitives
    absTensor :: a -> a
    addTensor :: a -> a -> a
    concatenate :: [a] -> Int -> a

class (StrictTensor a) => AutomaticTensor a where
    promoteToShape :: a -> [Int] -> a
    demoteToShape :: a -> [Int] -> a

rankOf :: (BehavesLikeATensor a b) => a b -> Int
rankOf = length . shape

tensorCopyWithNewName :: (BehavesLikeATensor a b) => a b -> String -> a b
tensorCopyWithNewName = renameTensor

showTensor :: (BehavesLikeATensor a b) => a b -> String
showTensor t = nameOf t ++ "[" ++ show (shape t) ++ "]"

instance TensorType Blx.TensorType where
    symbol = show

-- instance BehavesLikeATensor Blx.BlxTensor Blx.TensorType where
