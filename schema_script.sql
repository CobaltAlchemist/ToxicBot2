-- -----------------------------------------------------
-- Table `users`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `users` (
  `guild` INTEGER NOT NULL,
  `author` INTEGER NOT NULL,
  `incidents` INTEGER DEFAULT 0,
  `total_seen` INTEGER DEFAULT 0,
  `toxic` INTEGER DEFAULT 0,
  `severe_toxic` INTEGER DEFAULT 0,
  `obscene` INTEGER DEFAULT 0,
  `identity_attack` INTEGER DEFAULT 0,
  `insult` INTEGER DEFAULT 0,
  `threat` INTEGER DEFAULT 0,
  `sexual_explicit` INTEGER DEFAULT 0,
  PRIMARY KEY (`guild`, `author`));


-- -----------------------------------------------------
-- Table `last_messages`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `last_messages` (
  `guild` INTEGER NOT NULL,
  `author` INTEGER NOT NULL,
  `channel` INTEGER NOT NULL,
  `text` LONGTEXT NULL,
  PRIMARY KEY (`guild`, `author`, `channel`));


-- -----------------------------------------------------
-- Table `history`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `history` (
  `id` INTEGER PRIMARY KEY NOT NULL,
  `guild` INTEGER NOT NULL,
  `author` INTEGER NOT NULL,
  `toxic` REAL,
  `severe_toxic` REAL,
  `obscene` REAL,
  `identity_attack` REAL,
  `insult` REAL,
  `threat` REAL,
  `sexual_explicit` REAL);
