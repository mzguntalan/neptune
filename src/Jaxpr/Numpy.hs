module Jaxpr.Numpy where

import Data.Maybe (isNothing)
import Jaxpr.Blx (BlxTrace)
import Prelude hiding (abs, all)

-- This module has automatic convertions to BlxPrimitives

type NumpyArray = BlxTrace

-- ndarray.at
abs :: NumpyArray -> NumpyArray
abs = absolute

absolute :: NumpyArray -> NumpyArray
absolute x = _

acos :: NumpyArray -> NumpyArray
acos x = _

acosh :: NumpyArray -> NumpyArray
acosh x = _

add :: NumpyArray -> NumpyArray -> NumpyArray
add x y = _

all :: NumpyArray -> Maybe Int -> Maybe Bool -> Maybe NumpyArray -> NumpyArray
all a axis keepdims whereArray
    | isNothing keepdims = all a axis (Just False) whereArray
    | otherwise = _

allClose :: NumpyArray -> NumpyArray -> Maybe Float -> Maybe Float -> Maybe Bool -> NumpyArray
allClose a b rtol atol equalNan
    | isNothing rtol = allClose a b (Just 0.000_01) atol equalNan
    | isNothing atol = allClose a b rtol (Just 0.000_000_01) equalNan
    | isNothing equalNan = allClose a b rtol atol (Just False)
    | otherwise = _
