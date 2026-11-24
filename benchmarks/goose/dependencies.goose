// SPDX-License-Identifier: LicenseRef-HyperL-Community-1.0
fn main() {
    let jobs = 4000000;
    var finish: i32[>..] = [];
    finish.push(1);
    var checksum = 1;
    for i in 1..jobs {
        let left = i - 1 - ((i * 4051 % 65521) % i);
        let right = i - 1 - ((i * 2654435761 % 4093) % i);
        var ready = finish[left];
        if finish[right] > ready { ready = finish[right]; }
        let end = ready + 1 + i % 13;
        finish.push(end as i32);
        checksum += end;
    }
    print(checksum);
}
