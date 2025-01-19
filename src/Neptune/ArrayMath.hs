{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# OPTIONS_GHC -Wall #-}

module Neptune.ArrayMath where

import GHC.TypeLits

-- idk
-- what if Neptune was a design oriented language ?
-- you would define behaviors and shapes until you figure out the implementation details
-- what kind of traits are important?
-- Array type shape
-- Positive, Negative, NonNegative, NonPositive
-- SumTo1
-- Between a b

data ArrayType = Af32 | Ai32 | Af16 deriving (Show)

type Axis = String

type Shape = [Axis]

data Array = Array ArrayType Shape deriving (Show)

af32 :: Shape -> Array
af32 = Array Af32

ai32 :: Shape -> Array
ai32 = Array Ai32

rank :: Array -> Int
rank (Array _ shape) = length shape

data Between (low :: Float) (high :: Float) a where
    Between :: Float -> Float -> a -> Between low high a
    deriving (Show)

clip :: Float -> Float -> Array -> Between a b Array
clip = Between

clipNum :: Float -> Float -> Float -> Between a b Float
clipNum a b n
    | n > b = Between a b b
    | n < a = Between a b a
    | otherwise = Between a b n

f :: Between a b Float -> String
f (Between a b _)
    | a >= 0.0 && b <= 1.0 = "Valid Range"
    | otherwise = "Not in valid Range"

mul :: Array -> Array -> Array
mul (Array Af32 [a, b]) (Array Af32 [c, d])
    | b == c = af32 [a, d]
    | otherwise = error "Shape incompatible"
mul (Array Af32 shape1) (Array Af32 shape2)
    | length shape1 /= length shape2 = error "Shape incompatible"
    | length shape1 > 2 = Array Af32 newshape
  where
    (a : (b : _)) = reverse shape1
    (c : (d : _)) = reverse shape2
    (Array _ newshape) = mul (Array Af32 [a, b]) (Array Af32 [c, d])
mul _ _ = error "Not implemented"
