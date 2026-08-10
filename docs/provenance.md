# Real-use provenance

The inventory emphasizes repetitive browser work reported by developers: inspecting
records that are exposed primarily through a rendered portal, extracting tables and
documents, submitting existing forms, reconciling related pages, and validating
bounded UI workflows. The task is browser-relevant even when a trusted benchmark
hook uses an API to seed or grade it.

## Published-form workflows

Tally documents public form sharing, [hidden fields](https://tally.so/help/hidden-fields),
[password-protected forms](https://tally.so/help/password-protect-forms), and
[file uploads](https://tally.so/help/file-upload). The six benchmark forms combine
those real respondent workflows with localization, PDF-to-form transcription, and
cross-field reconciliation. They do not ask the agent to create or design a form.

The API is restricted to trusted lifecycle code. The measured task remains submission
through the real public UI, including rendered validation and the password gate.

## Public portals and official simulators

- [ATO online services simulator](https://www.ato.gov.au/calculators-and-tools/ato-online-services-simulator)
- [FCC equipment authorization search](https://www.fcc.gov/oet/ea/fccid)
- [FDA 510(k) Premarket Notification](https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfpmn/pmn.cfm)
  and [medical-device recalls](https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfres/res.cfm)
- [CBP CROSS rulings search](https://rulings.cbp.gov/)
- [USDA Organic Integrity Database](https://organic.ams.usda.gov/integrity/)
- [IRS Tax Withholding Estimator](https://www.irs.gov/individuals/tax-withholding-estimator)
- [Federal Student Aid Repayment Calculator](https://studentaid.gov/repayment-calculator/)
- [GOV.UK visa checker](https://www.gov.uk/check-uk-visa),
  [holiday entitlement](https://www.gov.uk/calculate-your-holiday-entitlement), and
  [redundancy pay](https://www.gov.uk/calculate-your-redundancy-pay)

These targets preserve real differences in HTML, accessibility markup, filtering,
tables, pagination, branching flows, generated results, downloads, and documents.
`--parallel` caps overall concurrency; ATO simulator tasks run sequentially. The
catalog does not submit legal applications or contact real people.

## Controlled complement

Five controlled tasks (5% of the 100-task catalog) provide deterministic coverage
for flows that are difficult to make safe and repeatable on public services: OTP
authentication, role permissions, paginated exports, upload validation, and
optimistic-concurrency conflicts. Each uses an independent loopback process and
trusted cleanup; the other catalog tasks use real public interfaces.

## Explicit exclusions

- Payments, purchases, bookings, legal applications, and external communication.
- Open-ended site, form, dashboard, report, board, or workflow construction.
- Tasks whose browser step is merely a less efficient replacement for a comprehensive
  supported API, such as ordinary Slack, Stripe, or PayPal operations.
- CAPTCHA-gated targets, volatile inventory, and access to non-benchmark user data.
