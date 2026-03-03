# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## Key Changes from Last 88 Commits (2024-11-04 to 2025-02-24)

### Added

- New API endpoints and refactored endpoint naming (#34) (`a4e3fdbb`, `6e871d95`)
- Improved dataset management with new download, list, and delete endpoints (#33) (`56977db3`)
- User developer guides for onboarding and documentation (#31) (`0c14f569`)
- New version of Nebula attacks system with refactored attack architecture (#30) (`377751a7`)
- Centralized databases for improved consistency across the platform (`6fee3155`)
- Improved scenario management with simultaneous queues support (#29) (`aebcbbb2`)
- Statistics metrics collection and reporting (`7662a110`)
- Ability to add new scenarios to the running queue (`5b101a7e`)
- Toggle for enabling/disabling report status in frontend (`24ba2b22`)
- Relaunch button on the dashboard for re-running scenarios (`21a2d51f`)
- Tooltips for deployment and dashboard buttons (`c3847b29`, `8ba351a7`)
- Show scenario configuration details via config button (`72aaee1e`)
- Updated icons for uploading and downloading scenarios (`dde22577`)
- Auto-update graph nodes based on user input, replacing random topology generation (`79b00ed6`)
- Frontend persistence for session state (`7cca67f4`)

### Changed

- Optimization of messages and communication (#32) (`cd002d7e`)
- Updated random topology generation logic (`ca2c94ed`, `812b3a4f`, `ab7d165f`)
- GPU option removed from frontend interface (`5fdb4ba2`)
- Removed step 11 from deployment workflow (`4b2a772a`)
- Changed `resources_threshold` from 40 to 80 for better resource management (`05076f9c`)
- Set predefined topology as default and display selection on graph (`6028f8b8`)
- Default option set to Docker containers in scenario creation (`6ffc9110`)
- Start new scenario with previous configuration by default (`e3ca9d3f`)
- Default number of rounds changed to 3 (`bf2796c5`)
- Moved number of rounds field to step 3 of deployment (`17a4a92c`)
- Renamed `poisoned_persent` to `poisoned_percent` for correctness (`4736c7e0`)
- Refactored topology manager functions with code descriptions (`6ea274c5`)
- Refactored poisoning attack code for improved type checking and readability (`70de2439`)
- Improved metric names in reporter and lightning modules (`35aa95ad`)
- Improved tooltips with more descriptive information (`0b7816ac`)
- Launch scenario with default options when no scenario is saved (`d2e0316e`)
- Changed scenario name labels to scenario titles (`ff9ec9ad`)
- Reordered elements in statistics page (`6fb382a1`)
- Removed unnecessary console logs (`be0edb93`)
- Dashboard reload removed (`b900fabc`)
- Relaxed documentation build warnings (`0fef7212`)

### Fixed

- Port assignment and port definition issues (`d4283bd1`, `fedcd9ca`)
- Improved error messages for resource limitations and WebSocket issues (`ac6adaef`)
- Selection issue with processes (`3e8353eb`)
- Custom databases path handling (`b2b8cc1a`)
- Correctly retrieve attack parameters from frontend (`697cd4f5`)
- Label-flipping attack implementation (`f61351a3`)
- Grafana dashboard configuration (`ee355611`)
- Stopping all scenarios when not enough resources, plus WebSocket errors (`bc7126d5`)
- Monitor page not loading (`cbf6d928`)
- Free frontend port discovery and stop containers by username (`155fd08e`)
- Timeout on async locks (`0ee87a82`, `241c7708`)
- Register callback (`60c91ef3`)
- Infinite scenarios loop bug (`5f101066`)
- Relaunch bug (`d425b840`)
- Load scenarios functionality (`558d0b2c`)
- Deployment process (`4c146dfd`)
- Loki configuration after update (`8ac31188`)
- Docker stop command (`ef0bdbb7`)
- Process launch compatibility with Python versions other than 3.11 (`36580b40`)

### Infrastructure

- Major changes in dependency management, docs, linting, and checks (`78a08e10`, `953c5188`)
- Updated Makefile and libraries (`9f7dc7a6`, `ef4b9138`, `9ca65942`)
- Improved installation process and removed unused code (`8f0ff6a7`)
- Updated GitHub Actions workflows (`03641c8c`, `e221e78a`, `871a436d`, `00e7a0f4`, `46c0cc92`)
- Updated virtual environment and Loki dependencies (`4fb92b07`)
- Changed dependencies location (`4a9a2ce2`)
- Removed keyring dependency from the project (`9f0df061`, `47fcb349`)
- Updated bootstrapping process (`5b44d54e`)
- Changed repository references to new location (`8da2e75d`)
- Updated project metadata (`8d89d7d9`)
- Removed outdated functions (`629d6fe9`)
- Updated Dockerfile and Docker configurations (`9d6cedc6`)

### Documentation

- Updated documentation across multiple areas (`e63b9c13`, `eba93124`, `83c2f2f5`, `408ab6e4`)
