// SPDX-License-Identifier: LicenseRef-HyperL-Community-1.0
fn main() {
    const JOBS: i64 = 4_000_000;
    let mut finish = Vec::<i32>::with_capacity(JOBS as usize);
    finish.push(1);
    let mut checksum = 1i64;
    for i in 1..JOBS {
        let left = i - 1 - ((i * 4051 % 65521) % i);
        let right = i - 1 - ((i * 2654435761 % 4093) % i);
        let ready = finish[left as usize].max(finish[right as usize]);
        let end = i64::from(ready) + 1 + i % 13;
        finish.push(end as i32);
        checksum += end;
    }
    println!("{checksum}");
}
