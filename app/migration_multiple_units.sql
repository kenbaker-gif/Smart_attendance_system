-- Migration to support multiple course unit assignments for coordinators
-- This changes course_unit_id from text to text[] (array)

-- First, add a temporary column to store the array version
ALTER TABLE profiles ADD COLUMN course_unit_ids text[];

-- Convert existing single values to arrays
UPDATE profiles
SET course_unit_ids = CASE
  WHEN course_unit_id IS NOT NULL AND course_unit_id != ''
  THEN ARRAY[course_unit_id]
  ELSE NULL
END
WHERE role = 'coordinator';

-- Drop the old column
ALTER TABLE profiles DROP COLUMN course_unit_id;

-- Rename the new column
ALTER TABLE profiles RENAME COLUMN course_unit_ids TO course_unit_id;

-- Add comment for documentation
COMMENT ON COLUMN profiles.course_unit_id IS 'Array of course unit IDs assigned to this coordinator';