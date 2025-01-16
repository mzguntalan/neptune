{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# OPTIONS_GHC -Wall #-}
{-# OPTIONS_GHC -Wno-unrecognised-pragmas #-}

{-# HLINT ignore "Redundant bracket" #-}

module Neptune.ArrayMath where

import Jaxpr (Shape, VarType)

-- all arrays are currently flat
data NeptuneArray a = NeptuneArray [a] Shape VarType

-- idk
data NaturalArray a = Array [a] | Collection [NaturalArray a]

type Axis = Int

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
shapeConcat (shape : shapes) axis = foldl (\x y -> shapeSingleStack x y axis) shape shapes
shapeConcat [] _ = error "no shapes to concat"

shapeSingleStack :: Shape -> Shape -> Axis -> Shape
shapeSingleStack (a : as) (b : bs) axis = f (a : as) (b : bs) axis 0
  where
    f :: Shape -> Shape -> Axis -> Axis -> Shape
    f [] [] target cur = if target == cur then [2] else error "not enough axes"
    f (c : cs) (d : ds) target cur
        | target == cur =
            if cs == ds && c == d
                then (2 : (c : cs))
                else error "shape mismatch"
        | target > cur =
            if (c : cs) == (d : ds)
                then c : f cs ds target (cur + 1)
                else error "shape mismatch"
        | otherwise = error "Not enough dimensions"
    f _ _ _ _ = error "Shouldn't Happen"
shapeSingleStack _ _ _ = error "Shape should be non empty list"

shapeStack :: [Shape] -> Axis -> Shape
shapeStack (shape : shapes) axis = foldl (\x y -> shapeSingleStack x y axis) shape shapes
shapeStack [] _ = error "no shapes to stack"
