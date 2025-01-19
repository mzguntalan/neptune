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

-- data Array =  deriving (Show)

af32 :: Shape -> Expression
af32 = Array Af32

ai32 :: Shape -> Expression
ai32 = Array Ai32

-- but we need to create expressions instead of evaluating them

data Expression = Array ArrayType Shape | Prim NeptunePrimitive [Expression] deriving (Show)

data NeptunePrimitive = NeptMul | NeptAdd deriving (Show)

mulExp :: Expression -> Expression -> Expression
mulExp a b = Prim NeptMul [a, b]
