# Find data regions in random data

> [!IMPORTANT]
> Copyright 2024, Clumio Inc.
> Licensed under the Apache License, Version 2.0 (the "License");
> you may not use this file except in compliance with the License.
> You may obtain a copy of the License at
>    http://www.apache.org/licenses/LICENSE-2.0
> Unless required by applicable law or agreed to in writing, software
> distributed under the License is distributed on an "AS IS" BASIS,
> WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
> See the License for the specific language governing permissions and
> limitations under the License.

## What is Find Data
This script will find the "signal" in the noise.  It uses an algorithim to identify significant data amoung the clutter and will return just the significant data

## How to use a Bulk Restore
An input file allows you to create a signal inside of a random data set for the process then to find.  You could also just point the algorithim at your own data.
The intent is that the solution is looking at time ordered data where you are counting events that have happened in a time set.  I.E. You cant have a negative events
If your data has negative values.  You would have to normalize the data.

## Example Use Case
Counting the number of time the refrigurator was opened in any given minute to determine when major meals happened.  This would help determine how many major meals occurred
durring the time span data was captured and the start and stop time for those meals.
