# Full telecom network-stack research plan

The owner selected full network-stack research, beyond AI traffic over cellular
links. Current code is a typed plan validator only; no carrier/RAN/core stack,
transmitter, SIM/subscriber credential service or standards certification exists.

Research layers: LTE/EPC and NR/5GC architecture; PHY DSP and FEC; MAC scheduling;
RLC/PDCP/SDAP; RRC/NAS state machines; control/user planes and GTP-U; authenticated
service interfaces; timing/synchronization, QoS/slicing; security/auth/key lifecycle;
MEC/AI placement, telemetry and bounded human/agent control. Each is independently
versioned and tested. HyperL kernels may accelerate mathematical workloads, but a
kernel engine does not implement or certify a network protocol stack.

Primary standards watchlist:

- [3GPP RAN#113 reports](https://www.3gpp.org/news-events/3gpp-news/ran113-reports)
  discuss Release-20 5G-Advanced work and 6G studies. Treat future requirements as
  research, not a final universal 6G compliance target.
- [3GPP TS 33.501](https://portal.3gpp.org/desktopmodules/Specifications/SpecificationDetails.aspx?specificationId=3169)
  for 5G security architecture; pin exact editions and required procedures per lab.
- [ETSI TS 129 281 / 3GPP TS 29.281 V19.2.0](https://www.etsi.org/deliver/etsi_ts/129200_129299/129281/19.02.00_60/ts_129281v190200p.pdf)
  for GTP-U. No parser/transport compatibility is claimed by this release.
- [ETSI MEC](https://labs.etsi.org/rep/mec) for independently versioned AI/edge
  orchestration interfaces, not a substitute for the radio/core standards.
- [ISO/IEC 27001:2022](https://www.iso.org/standard/27001) concerns an organization's
  information security management system; a software flag cannot confer that
  certification or prove telecom interoperability.

Optional stack research candidates: [OpenAirInterface](https://openairinterface.org/)
and [srsRAN](https://www.srsran.com/). Review exact repository/component editions,
licenses and [OAI's component license model](https://openairinterface.org/oai-license-model/)
before any integration. No stack source, simulator, driver or RF package is imported.

Build order: offline bounded message/DSP fixtures → reference state-machine tests
and malformed-input/cancellation cases → isolated simulated core/RAN lab → private
adapter conformance and security tests → qualified hardware/timing/failure tests.
Actual radio work needs a separately authorized, isolated test environment and
applicable spectrum authorization. A profile boolean never grants those privileges.
Maintain traceability from exact standard clause/version to implementation/test,
record unsupported extensions, and require independent review before production use.
