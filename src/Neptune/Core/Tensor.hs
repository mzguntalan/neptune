{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# OPTIONS_GHC -Wno-unrecognised-pragmas #-}

module Neptune.Core.Tensor where

import Data.List (intercalate)
import Neptune.Core.Program

data OutputDescription a = (Eq a) => OutputDescription a

instance Eq (OutputDescription a) where
    (OutputDescription a1) == (OutputDescription a2) = a1 == a2

data Zprim a = Zprim String [(String, String)] (OutputDescription a) deriving (Eq)

instance Show (Zprim a) where
    show (Zprim name params _) = name ++ "[" ++ intercalate "," (map pairShow params) ++ "]"
      where
        pairShow (x, y) = x ++ "=" ++ y

type ZprimInstruction a = PrimitiveInstruction (Zprim a)

type Shape = [Int]

type Tensor = Program (Zprim Shape)

tensorOf :: Shape -> Tensor
tensorOf shape = Immediate prim []
  where
    prim = PrimitiveInstruction (Zprim "create" [("shape", show shape)] (OutputDescription shape))

getShape :: Tensor -> Shape
getShape t = s
  where
    PrimitiveInstruction (Zprim _ _ (OutputDescription s)) = lastPrimitiveInstructionInProgram t

addTensor :: Tensor -> Tensor -> Tensor
addTensor a b
    | getShape a == getShape b = Immediate prim [a, b]
    | otherwise = error "either autobroadcast, or give up"
  where
    prim = PrimitiveInstruction (Zprim "add" [] (OutputDescription (getShape a)))

absTensor :: Tensor -> Tensor
absTensor a = Immediate (PrimitiveInstruction (Zprim "abs" [] (OutputDescription (getShape a)))) [a]

broadcastInDim :: (Show a) => a -> Shape -> Tensor
broadcastInDim val shape = Immediate (PrimitiveInstruction (Zprim "broadcast_in_dim" [("dim", "[]"), ("shape", show shape), ("val", show val)] (OutputDescription shape))) []

allEq :: (Eq a) => [a] -> Bool
allEq [] = True
allEq [_] = True
allEq (x : (y : others)) = x == y && allEq others

shapeAllEq :: [Tensor] -> Bool
shapeAllEq ts = allEq shapes
  where
    shapes = map getShape ts

sumTensors :: [Tensor] -> Tensor
sumTensors (t : ts)
    | shapeAllEq (t : ts) = Immediate (PrimitiveInstruction (Zprim "sum" [] (OutputDescription (getShape t)))) (t : ts)
    | otherwise = error "shape error"
sumTensors [] = error "empty"

ones :: Shape -> Tensor
ones = broadcastInDim (1.0 :: Float)

zeros :: Shape -> Tensor
zeros = broadcastInDim (0.0 :: Float)
