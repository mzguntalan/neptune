-- {-# LANGUAGE DataKinds #-}
-- {-# LANGUAGE FlexibleInstances #-}
-- {-# LANGUAGE GADTs #-}
-- {-# LANGUAGE MultiParamTypeClasses #-}

module Jaxpr.Compiler where

-- import Data.List (findIndex, intercalate, nub, nubBy)
-- import Data.Map.Strict qualified as Map
-- import Jaxpr.Blx.Primitives
-- import Jaxpr.Blx.Tensor
-- import Jaxpr.Blx.Trace

-- data JaxExpression = JaxExpression [BlxTensor] [BlxTensor] [TraceEntry] [BlxTensor]

-- mkJaxExpression :: [BlxTensor] -> [BlxTensor] -> [TraceEntry] -> [BlxTensor] -> JaxExpression
-- mkJaxExpression litInputs varInputs traceentries outputs = JaxExpression litInputs varInputs traceentries outputs

-- instance Show JaxExpression where
--     show (JaxExpression consts inVars entries outs) = "{ lambda " ++ constShow ++ " ; " ++ inVarsShow ++ ". let\n\t" ++ entriesShow ++ "\n in " ++ outVarsShow ++ " }"
--       where
--         constShow = unwords (map show consts)
--         inVarsShow = unwords (map show inVars)
--         entriesShow = intercalate "\n\t" (map show entries)
--         outVarsShow = "(" ++ intercalate "," (map tensorName outs) ++ ",)"

-- compileToJaxpr :: BlxTrace -> JaxExpression
-- compileToJaxpr = prettifyJaxpr . traceToJaxExpression

-- prettifyJaxpr :: JaxExpression -> JaxExpression
-- prettifyJaxpr (JaxExpression consts inVars entries outs) = JaxExpression renamedConsts renamedInVars renamedEntries renamedOuts
--   where
--     allVars :: [BlxTensor]
--     allVars = consts ++ inVars ++ concatMap entriesAllVars entries ++ outs
--     allVarNames = nub $ map tensorName allVars
--     varnamePool = map varNameFromInt [0, 1 .. (length allVars - 1)]
--     lookupOfVarNames :: Map.Map String String
--     lookupOfVarNames = Map.fromList (zip allVarNames varnamePool)

--     sameName :: BlxTensor -> BlxTensor -> Bool
--     sameName t1 t2 = tensorName t1 == tensorName t2
--     renamedConsts = nubBy sameName $ map (`renameTensorUsingMap` lookupOfVarNames) consts
--     renamedInVars = nubBy sameName $ map (`renameTensorUsingMap` lookupOfVarNames) inVars
--     renamedEntries = map (`renameEntriesUsingMap` lookupOfVarNames) entries
--     renamedOuts = map (`renameTensorUsingMap` lookupOfVarNames) outs

-- traceToJaxExpression :: BlxTrace -> JaxExpression
-- traceToJaxExpression tr = JaxExpression consts inVars entries outVars
--   where
--     allTraceEntries = reverse . traceEntries $ tr
--     entries = filter isJaxExpressionEntry allTraceEntries
--     isJaxExpressionEntry :: TraceEntry -> Bool
--     isJaxExpressionEntry x = not (isLitEntry x) && not (isVarEntry x)

--     litEntries = filter isLitEntry allTraceEntries
--     varEntries = filter isVarEntry allTraceEntries

--     consts = map (head . entryOutputs) litEntries
--     inVars = map (head . entryInputs) varEntries

--     outVars = currentTraceOutputs tr

-- renameTensorUsingMap :: BlxTensor -> Map.Map String String -> BlxTensor
-- renameTensorUsingMap t m = case Map.lookup (tensorName t) m of
--     Just newName -> renameTensor t newName
--     Nothing -> error "Name not found in lookup"

-- renameEntriesUsingMap :: TraceEntry -> Map.Map String String -> TraceEntry
-- renameEntriesUsingMap (TraceEntry prim inputs outputs) varmap = TraceEntry prim renamedInputs renamedOutputs
--   where
--     f :: BlxTensor -> BlxTensor
--     f t = renameTensor t newName
--       where
--         newName = case Map.lookup (tensorName t) varmap of
--             Just a -> a
--             Nothing -> error "You made a mistake with generating the correspondences of oldnames to new names"

--     renamedInputs = map f inputs
--     renamedOutputs = map f outputs

-- entriesAllVars :: TraceEntry -> [BlxTensor]
-- entriesAllVars entry = entryInputs entry ++ entryOutputs entry

-- symbolsForNaming :: String
-- symbolsForNaming = "abcdefghijklmnopqrstuvwxyz"

-- -- this needs fixing
-- varNameFromInt :: Int -> String
-- varNameFromInt x
--     | 0 <= x && x < length symbolsForNaming = [symbolsForNaming !! x]
--     | x >= length symbolsForNaming = "a" ++ varNameFromInt (x - length symbolsForNaming)
--     | otherwise = error "Shouldn't happen"

-- isVarEntry :: TraceEntry -> Bool
-- isVarEntry (TraceEntry prim _ _) = prim == TracePrimitive Var

-- isLitEntry :: TraceEntry -> Bool
-- isLitEntry (TraceEntry prim _ _) = prim == TracePrimitive Lit
