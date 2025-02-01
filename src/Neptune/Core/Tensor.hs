{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# OPTIONS_GHC -Wno-unrecognised-pragmas #-}

{-# HLINT ignore "Use newtype instead of data" #-}

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
