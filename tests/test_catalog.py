from __future__ import annotations

import json
import unittest
from collections import Counter

from rbbench.catalog import BENCHMARK_TASK_IDS, REPO_ROOT, load_catalog


class CatalogTests(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = load_catalog()

    def test_catalog_has_expected_real_environment_mix(self) -> None:
        self.assertEqual(len(self.catalog.tasks), 100)
        self.assertEqual(
            [task.task_id for task in self.catalog.tasks],
            list(BENCHMARK_TASK_IDS),
        )
        self.assertEqual(
            Counter(task.environment.adapter for task in self.catalog.tasks),
            {
                "ato_simulator": 9,
                "tally_public_form": 6,
                "public_web": 80,
                "controlled_portal": 5,
            },
        )
        synthetic = [
            task
            for task in self.catalog.tasks
            if "synthetic" in task.environment.kind
        ]
        self.assertEqual(len(synthetic), 5)

    def test_every_mutable_task_requires_verified_cleanup(self) -> None:
        mutable = [task for task in self.catalog.tasks if task.environment.mutable]
        self.assertEqual(len(mutable), 11)
        self.assertTrue(all(task.cleanup.verify_absence for task in mutable))
        self.assertTrue(
            all("cleanup" in task.environment.required_hooks for task in mutable)
        )

    def test_tally_tasks_include_attempt_isolation_and_login_variants(self) -> None:
        tally = [
            task
            for task in self.catalog.tasks
            if task.environment.adapter == "tally_public_form"
        ]
        self.assertEqual(len(tally), 6)
        self.assertTrue(
            all(
                task.fixture["expected_submission"]["attempt_id"] == "{{attempt_id}}"
                and task.fixture["expected_submission"]["task_id"] == task.task_id
                for task in tally
            )
        )
        self.assertEqual(
            [task.task_id for task in tally if task.environment.auth == "form_password"],
            ["RBA-010", "RBA-012"],
        )
        self.assertTrue(
            all(
                task.environment.concurrency_key == "tally.so"
                and task.environment.concurrency_limit == 2
                for task in tally
            )
        )

    def test_local_input_artifacts_exist(self) -> None:
        for task in self.catalog.tasks:
            for key in ("input_artifact", "body_artifact", "content_artifact"):
                if key in task.fixture:
                    path = REPO_ROOT / str(task.fixture[key])
                    self.assertTrue(path.exists(), f"{task.task_id}: missing {path}")

    def test_browser_task_text_contains_attempt_scope_when_mutable(self) -> None:
        for task in self.catalog.tasks:
            if task.environment.mutable:
                self.assertIn("{{attempt_id}}", task.confirmed_task, task.task_id)

    def test_download_tasks_do_not_require_agent_side_renaming(self) -> None:
        for task in self.catalog.tasks:
            if "download" not in task.category and "document_download" not in task.category:
                continue
            self.assertNotIn(" as RBA-", task.confirmed_task, task.task_id)
            for assertion in task.oracle.assertions:
                if assertion.kind == "artifact_matches":
                    expected = assertion.expected or {}
                    self.assertNotIn("name", expected, task.task_id)

    def test_explicit_text_only_forms_do_not_mention_uploads(self) -> None:
        explicit = load_catalog(REPO_ROOT / "tasks" / "tasks-v2.json")
        text_only_form_ids = {
            "RBA-073",
            "RBA-074",
            "RBA-075",
            "RBA-076",
            "RBA-078",
            "RBA-079",
            "RBA-081",
            "RBA-082",
            "RBA-085",
            "RBA-086",
            "RBA-087",
            "RBA-090",
            "RBA-092",
            "RBA-093",
            "RBA-094",
            "RBA-095",
        }
        self.assertEqual(len(text_only_form_ids), 16)
        for task_id in text_only_form_ids:
            prompt = explicit.by_id(task_id).confirmed_task
            self.assertTrue(
                "Short answer" in prompt or "Long answer" in prompt or "Paragraph" in prompt,
                task_id,
            )
            self.assertNotIn("uploaded filename", prompt, task_id)
            self.assertNotIn("upload_files", prompt, task_id)

    def test_explicit_tally_forms_use_visible_editor_navigation(self) -> None:
        explicit = load_catalog(REPO_ROOT / "tasks" / "tasks-v2.json")
        for index in range(72, 84):
            task_id = f"RBA-{index:03d}"
            prompt = explicit.by_id(task_id).confirmed_task
            for visible_instruction in (
                "New form",
                "Blank form",
                "Click to insert your first block",
                "Insert anything",
                "click Insert",
                "Type a question",
                "required",
                "Publish",
                "Submissions",
            ):
                self.assertIn(visible_instruction, prompt, task_id)
            self.assertNotIn("Repeat that sequence", prompt, task_id)

    def test_explicit_google_forms_reuse_the_seed_question(self) -> None:
        explicit = load_catalog(REPO_ROOT / "tasks" / "tasks-v2.json")
        single_question_ids = {"RBA-085", "RBA-090", "RBA-094"}
        three_additions_id = "RBA-087"
        for index in range(84, 96):
            task_id = f"RBA-{index:03d}"
            prompt = explicit.by_id(task_id).confirmed_task
            self.assertIn("existing Untitled Question", prompt, task_id)
            self.assertIn("Publish", prompt, task_id)
            self.assertIn("Responses", prompt, task_id)
            if task_id in single_question_ids:
                self.assertIn("Do not click Add question", prompt, task_id)
            elif task_id == three_additions_id:
                self.assertIn("Click Add question three times", prompt, task_id)
            else:
                self.assertRegex(prompt, r"Click Add question (exactly )?once", task_id)

    def test_explicit_box_tasks_use_observed_file_and_note_controls(self) -> None:
        explicit = load_catalog(REPO_ROOT / "tasks" / "tasks-v2.json")
        upload_ids = {
            *[f"RBA-{index:03d}" for index in range(40, 46)],
            *[f"RBA-{index:03d}" for index in range(51, 60)],
        }
        note_ids = {f"RBA-{index:03d}" for index in range(60, 72)}
        for task_id in upload_ids:
            prompt = explicit.by_id(task_id).confirmed_task
            self.assertIn("https://app.box.com/", prompt, task_id)
            self.assertIn("upload_files", prompt, task_id)
            self.assertIn("File Upload", prompt, task_id)
            self.assertRegex(
                prompt.lower(),
                r"(?:do not click|without clicking) file upload first",
                task_id,
            )
        for task_id in note_ids:
            prompt = explicit.by_id(task_id).confirmed_task
            self.assertIn("https://app.box.com/", prompt, task_id)
            self.assertIn("Box Note", prompt, task_id)
            self.assertIn("Untitled", prompt, task_id)
            self.assertIn("Start writing", prompt, task_id)
            self.assertIn("All changes saved", prompt, task_id)

    def test_sputnik_form_uses_a_concrete_image_download_route(self) -> None:
        explicit = load_catalog(REPO_ROOT / "tasks" / "tasks-v2.json")
        prompt = explicit.by_id("RBA-083").confirmed_task
        for visible_instruction in (
            "Sputnik 1's launch date",
            "Wikimedia Commons",
            "Sputnik 1.jpg",
            "File:Sputnik 1.jpg",
            "Original file",
            "download the displayed JPEG",
            "Sputnik_1.jpg",
            "before opening Tally",
        ):
            self.assertIn(visible_instruction, prompt)
        self.assertNotIn("Research the requested facts", prompt)

    def test_explicit_image_forms_name_fact_and_image_routes(self) -> None:
        explicit = load_catalog(REPO_ROOT / "tasks" / "tasks-v2.json")
        expected_routes = {
            "RBA-072": [
                "https://programs.wcs.org/china/Wildlife/Amur-Leopard.aspx",
                "Status section",
                "commons.wikimedia.org/wiki/File:Amur_Leopard",
                "Original file",
            ],
            "RBA-077": [
                "https://www.nps.gov/yell/planyourvisit/parkfacts.htm",
                "History",
                "NPS Old Faithful photo page",
                "960-pixel",
            ],
            "RBA-080": [
                "jetstream/clouds/ten-basic-clouds",
                "Cumulonimbus (Cb)",
                "photo-library/wea00105jpg",
                "Download",
            ],
            "RBA-084": [
                "science.nasa.gov/asset/webb/exoplanet-kepler-186f-illustration",
                "About the Object",
                "Distance",
                "Downloads",
            ],
            "RBA-088": [
                "galleriaaccademiafirenze.it/en/artworks/david-michelangelo",
                "Data sheet, Dimensions",
                "commons.wikimedia.org/wiki/File:Michelangelos_David.jpg",
                "Original file",
            ],
            "RBA-089": [
                "burjkhalifa.ae/img/FACT-SHEET.pdf",
                "Height to Architectural Top",
                "commons.wikimedia.org/wiki/File:Burj_Khalifa_panorama_view.jpg",
                "Original file",
            ],
            "RBA-091": [
                "worldathletics.org/competitions/world-athletics-championships",
                "Usain Bolt's Mark",
                "commons.wikimedia.org/wiki/File:Boltarrivee200.jpg",
                "crossing the finish line",
                "Original file",
            ],
        }
        self.assertEqual(len(expected_routes), 7)
        for task_id, tokens in expected_routes.items():
            prompt = explicit.by_id(task_id).confirmed_task
            for token in tokens:
                self.assertIn(token, prompt, task_id)
            self.assertIn("upload_files", prompt, task_id)
            self.assertNotIn("Research the requested facts", prompt, task_id)
            self.assertNotIn("requested image", prompt, task_id)

    def test_explicit_text_forms_name_source_and_extraction_path(self) -> None:
        explicit = load_catalog(REPO_ROOT / "tasks" / "tasks-v2.json")
        expected_routes = {
            "RBA-073": ["asc-csa.gc.ca/eng/astronomy/solar-system/jupiter.asp", "Jupiter fact table", "Average distance from the Sun"],
            "RBA-074": ["pubchem.ncbi.nlm.nih.gov/element/Bismuth", "Atomic Number", "Atomic Mass"],
            "RBA-075": ["nobelprize.org/prizes/physics/2025/summary", "first displayed laureate", "shared prize citation"],
            "RBA-076": ["cms.cern/news/cms-precisely-measures-mass-higgs-boson", "highlighted measured", "GeV"],
            "RBA-078": ["docs.python.org/3/faq/general.html", "Why was Python created in the first place?", "first public USENET release"],
            "RBA-079": ["metzdowd.com/pipermail/cryptography/2008-October/014810.html", "message header", "Bitcoin: A Peer-to-Peer Electronic Cash System"],
            "RBA-081": ["ourworldindata.org/grapher/solar-energy-generation-by-region", "latest displayed year", "TWh"],
            "RBA-082": ["doae.go.th/en/pad-thai", "Ingredients section", "optional garnishes"],
            "RBA-085": ["home.nps.gov/maca/learn/management/statistics.htm", "Total Length of Mammoth Cave (to date)"],
            "RBA-086": ["chateauversailles.fr/discover/history/key-dates/treaty-versailles-1919", "opening account", "named room"],
            "RBA-087": ["learn.microsoft.com/en-us/dotnet/csharp/fundamentals/tutorials/oop", "four basic object-oriented principles"],
            "RBA-090": ["escoffier.edu/blog/recipes/how-to-make-the-five-mother-sauces", "five named sauce sections"],
            "RBA-092": ["cisa.gov/news-events/ics-advisories/icsa-10-272-01", "Overview", "Affected Products"],
            "RBA-093": ["shakespearesglobe.com/discover/shakespeares-world/the-globe", "When and where was the Globe built?"],
            "RBA-094": ["rgs.org/our-collections/buy-and-license-images/platinum-prints/everest-1953", "opening account"],
            "RBA-095": ["airandspace.si.edu/stories/editorial/breaking-sound-barrier", "opening account"],
        }
        self.assertEqual(len(expected_routes), 16)
        for task_id, tokens in expected_routes.items():
            prompt = explicit.by_id(task_id).confirmed_task
            for token in tokens:
                self.assertIn(token, prompt, task_id)
            self.assertNotIn("Research the requested facts", prompt, task_id)
            self.assertNotIn("upload_files", prompt, task_id)

    def test_explicit_box_notes_name_stable_sources_and_sections(self) -> None:
        explicit = load_catalog(REPO_ROOT / "tasks" / "tasks-v2.json")
        expected_routes = {
            "RBA-061": ["cloudflare.com/learning/ddos/glossary/user-datagram-protocol-udp", "How does UDP work?", "How is UDP different from TCP?"],
            "RBA-062": ["ats.aq/e/antarctictreaty.html", "Some important provisions"],
            "RBA-063": ["cdc.gov/covid/vaccines/how-they-work.html", "mRNA vaccines", "How mRNA COVID-19 vaccines work"],
            "RBA-064": ["web.stanford.edu/~peastman/statmech/thermodynamics.html", "5.2. The Laws of Thermodynamics", "Zeroth"],
            "RBA-066": ["cs.stanford.edu/people/eroberts/courses/soco/projects/risc/risccisc/index.html", "The CISC Approach", "RISC Roadblocks"],
            "RBA-067": ["britannica.com/science/tragedy-of-the-commons", "opening definition", "incentive mechanism", "environmental examples"],
            "RBA-068": ["kernel.org/doc/ols/2011/ols2011-masters.pdf", "historical overview", "2.x"],
            "RBA-069": ["imf.org/external/np/exr/center/mm/eng/cc_sub_4.htm", "Bretton Woods: July 1-22, 1944", "International Monetary Fund and World Bank"],
            "RBA-070": ["britannica.com/topic/bystander-effect", "Latané and Darley's experiments", "proposed mechanisms"],
            "RBA-071": ["pubmed.ncbi.nlm.nih.gov/28732554", "Abstract", "yeast and lactobacilli"],
        }
        self.assertEqual(len(expected_routes), 10)
        for task_id, tokens in expected_routes.items():
            prompt = explicit.by_id(task_id).confirmed_task
            for token in tokens:
                self.assertIn(token, prompt, task_id)
            self.assertNotIn("From a substantive", prompt, task_id)
            self.assertNotIn("substantive source", prompt, task_id)
            self.assertIn("https://app.box.com/", prompt, task_id)
        self.assertNotIn("find", explicit.by_id("RBA-068").confirmed_task.lower())

    def test_met_conversion_task_uses_available_jpeg_source(self) -> None:
        canonical = self.catalog.by_id("RBA-057")
        explicit = load_catalog(REPO_ROOT / "tasks" / "tasks-v2.json").by_id(
            "RBA-057"
        )
        self.assertEqual(canonical.fixture["source_format"], "JPEG")
        for task in (canonical, explicit):
            self.assertIn("JPEG", task.title)
            self.assertIn("JPEG", task.confirmed_task)
            self.assertNotIn("PNG", task.confirmed_task)

    def test_librivox_conversion_uses_explicit_ogg_to_wav_route(self) -> None:
        canonical = self.catalog.by_id("RBA-059")
        explicit = load_catalog(REPO_ROOT / "tasks" / "tasks-v2.json").by_id(
            "RBA-059"
        )
        self.assertEqual(canonical.fixture["source_format"], "OGG")
        self.assertEqual(canonical.fixture["output"], "WAV")
        self.assertEqual(canonical.fixture["converter"], "tinywow")
        self.assertNotIn("MP3", canonical.title)
        for source in (
            "https://librivox.org/pride-and-prejudice-v-4-by-jane-austen/",
            "https://archive.org/details/pride_prejudice_1102_librivox",
            "https://tinywow.com/video/ogg-to-wav",
        ):
            self.assertIn(source, canonical.sources)
        for visible_instruction in (
            "Pride and Prejudice (version 5)",
            "Internet Archive Page",
            "DOWNLOAD OPTIONS",
            "OGG VORBIS",
            ".ogg",
            "https://tinywow.com/video/ogg-to-wav",
            "Upload from PC or Mobile",
            ".wav",
            "upload_files",
        ):
            self.assertIn(visible_instruction, explicit.confirmed_task)
        self.assertNotIn("MP3", explicit.confirmed_task)
        self.assertNotIn("matching converter", explicit.confirmed_task)

    def test_fixed_target_tasks_expose_inputs_but_hide_reference_results(self) -> None:
        expected = {
            "RBA-096": {
                "fixture": {"scenario_number": 5, "income_year": "2025-26"},
                "prompt_tokens": ["scenario 5", "2025-26"],
                "fields": ["taxpayer", "employer", "gross_payment", "tax_withheld"],
                "hidden_primary": "Steve Isaacs",
            },
            "RBA-097": {
                "fixture": {
                    "grantee_code": "2AG32",
                    "product_code": "BSC7261A249D",
                },
                "prompt_tokens": ["2AG32", "BSC7261A249D"],
                "fields": [
                    "applicant",
                    "grant_purpose",
                    "grant_date",
                    "frequency_range",
                ],
                "hidden_primary": "Baicells Technologies Co., Ltd.",
            },
            "RBA-098": {
                "fixture": {"k_number": "K193299"},
                "prompt_tokens": ["K193299"],
                "fields": [
                    "device_name",
                    "applicant",
                    "decision_date",
                    "decision_type",
                ],
                "hidden_primary": "VITEK 2 AST-Gram Negative Ceftazidime",
            },
            "RBA-099": {
                "fixture": {"ruling_number": "J83798"},
                "prompt_tokens": ["J83798"],
                "fields": ["date", "subject", "tariff_classification"],
                "hidden_primary": "automotive seat structures from Canada",
            },
            "RBA-100": {
                "fixture": {"nop_id": "6903966799"},
                "prompt_tokens": ["6903966799"],
                "fields": [
                    "operation_name",
                    "certification_status",
                    "scopes",
                    "certified_acres",
                ],
                "hidden_primary": '"BREDUN" LP',
            },
        }
        for task_id, contract in expected.items():
            task = self.catalog.by_id(task_id)
            for key, value in contract["fixture"].items():
                self.assertEqual(task.fixture[key], value, task_id)
            for token in contract["prompt_tokens"]:
                self.assertIn(token, task.confirmed_task, task_id)
            self.assertEqual(task.fixture["fields"], contract["fields"], task_id)
            self.assertNotIn(contract["hidden_primary"], task.confirmed_task, task_id)

            reference_path = REPO_ROOT / "references" / "tasks" / f"{task_id}.json"
            reference = json.loads(reference_path.read_text(encoding="utf-8"))
            self.assertIn(contract["hidden_primary"], reference["result"]["primary"])

    def test_explicit_procedure_catalog_preserves_non_prompt_task_data(self) -> None:
        explicit = load_catalog(REPO_ROOT / "tasks" / "tasks-v2.json")
        tier_a_full_replacements = {
            "RBA-001",
            "RBA-002",
            "RBA-003",
            "RBA-005",
            "RBA-006",
            "RBA-008",
            "RBA-009",
            "RBA-010",
            "RBA-011",
            "RBA-012",
            "RBA-013",
            "RBA-014",
            "RBA-015",
            "RBA-018",
            "RBA-021",
            "RBA-024",
            "RBA-025",
            "RBA-026",
            "RBA-028",
            "RBA-031",
            "RBA-032",
            "RBA-033",
            "RBA-034",
            "RBA-046",
            "RBA-047",
            "RBA-048",
            "RBA-049",
            "RBA-050",
            "RBA-096",
            "RBA-097",
            "RBA-098",
            "RBA-100",
        }
        self.assertEqual(len(tier_a_full_replacements), 32)
        no_successful_route_tasks = {
            "RBA-041",
            "RBA-053",
            "RBA-054",
            "RBA-057",
            "RBA-059",
        }
        curated_replacement_ids = no_successful_route_tasks
        full_replacement_ids = ({
            f"RBA-{index:03d}" for index in range(1, 101)
        } - no_successful_route_tasks) | curated_replacement_ids
        remaining_route_replacements = full_replacement_ids - tier_a_full_replacements
        self.assertEqual(len(full_replacement_ids), 100)
        self.assertEqual(len(remaining_route_replacements), 68)
        full_replacement_routes = {
            "RBA-001": [
                "Prepare",
                "Contact details",
                "Income statements and payment summaries",
            ],
            "RBA-002": ["Manage tax returns", "History", "Original", "View details"],
            "RBA-003": ["Tax", "Lodgments", "Activity statement", "History"],
            "RBA-005": ["Higher education loan program (HELP)", "Accounts"],
            "RBA-006": ["Employment", "Employment details"],
            "RBA-007": [
                "Manage tax returns",
                "History",
                "Super",
                "Manage",
                "Non-concessional election",
            ],
            "RBA-008": ["My profile", "Communication", "History"],
            "RBA-009": ["Requester name", "Department", "Request type", "Submit"],
            "RBA-010": ["Password", "Continue", "Incident type", "Submit"],
            "RBA-011": ["Certificate file", "upload_files", "Expiry date", "Submit"],
            "RBA-012": ["Contraseña", "Continuar", "Producto", "Enviar"],
            "RBA-013": [
                "Keyboard quantity",
                "Requested total including tax",
                "Submit",
            ],
            "RBA-014": ["Vendor", "Invoice number", "Payment due", "Submit"],
            "RBA-015": ["Grantee Code", "Product Code", "Search"],
            "RBA-018": ["510K Number", "Search"],
            "RBA-021": ["Search", "Open ruling N296416"],
            "RBA-024": [
                "Advanced Search",
                "NOP Operation ID",
                "Scope and Product Summary",
            ],
            "RBA-025": [
                "Advanced Search",
                "NOP Operation ID",
                "Scope and Product Summary",
            ],
            "RBA-026": [
                "Advanced Search",
                "NOP Operation ID",
                "Export Operation Profile to PDF",
            ],
            "RBA-028": ["About you", "Income", "standard deduction", "Results"],
            "RBA-031": [
                "Start now",
                "Canada",
                "Tourism or visiting family and friends",
            ],
            "RBA-032": ["Start now", "India", "longer than 6 months"],
            "RBA-033": ["Start now", "days worked per week", "Outcome"],
            "RBA-034": ["Start now", "2026-07-31", "£720"],
            "RBA-046": ["Sign in", "Northstar Mail", "Return to sign in", "Account"],
            "RBA-047": [
                "VEN-204",
                "Download public attachment",
                "Edit record",
                "Permission denied",
            ],
            "RBA-048": [
                "Exception",
                "Apply filters",
                "Page 2",
                "Page 3",
                "Download CSV",
                "Download PDF",
            ],
            "RBA-049": [
                "CASE-1049",
                "upload_files",
                "Submit for validation",
                "Accepted",
            ],
            "RBA-050": ["EXC-550", "Save update", "Reload latest version"],
            "RBA-096": [
                "Prepare",
                "Contact details",
                "Income statements and payment summaries",
            ],
            "RBA-097": ["Grantee Code", "Product Code", "Search"],
            "RBA-098": ["510K Number", "Search"],
            "RBA-100": [
                "Advanced Search",
                "NOP Operation ID",
                "Scope and Product Summary",
            ],
        }
        full_replacement_routes.update(
            {
                "RBA-004": ["Tax", "Accounts", "Tax accounts", "Income tax 551"],
                "RBA-016": ["Grantee Code", "Product Code", "Exhibits"],
                "RBA-017": ["Grantee Code", "Product Code", "FCC E-Label Info"],
                "RBA-019": ["Medical Device Recalls", "Recall Number"],
                "RBA-020": ["Product Classification", "Product Code QAS"],
                "RBA-022": ["CBP CROSS", "Search", "first five rows"],
                "RBA-023": ["CBP CROSS", "Download", "Word .doc"],
                "RBA-027": ["About you", "Income", "standard deduction", "Results"],
                "RBA-029": [
                    "Get Started",
                    "Continue Without Logging In",
                    "Show Plans",
                    "Lowest Monthly Payment",
                ],
                "RBA-030": [
                    "Get Started",
                    "Continue Without Logging In",
                    "Show Plans",
                    "Lowest Total Paid",
                ],
                "RBA-035": ["Financials", "Quarterly", "Name box", "Formula bar"],
                "RBA-036": ["World Bank", "Name box", "Formula bar"],
                "RBA-037": ["USGS Latest Earthquakes", "Name box", "Formula bar"],
                "RBA-038": ["FRED", "UNRATE", "Name box", "Formula bar"],
                "RBA-039": ["Power Search", "Highway MPG", "Name box", "Formula bar"],
                "RBA-040": ["Rosetta", "high-resolution", "upload_files"],
                "RBA-041": [
                    "Smithsonian Open Access",
                    "Apollo 11 Command Module Columbia",
                    "Download Image",
                    "TIFF",
                    "upload_files",
                ],
                "RBA-042": ["arXiv", "Download PDF", "upload_files"],
                "RBA-043": ["arXiv", "Download PDF", "upload_files"],
                "RBA-044": ["LibriVox", "first chapter", "upload_files"],
                "RBA-045": ["The Met", "Open Access", "upload_files"],
                "RBA-051": ["Thingiverse", "STL", "upload_files"],
                "RBA-052": ["NOAA Photo Library", "high-resolution", "upload_files"],
                "RBA-053": [
                    "https://www.jamendo.com/start",
                    "instrumental and acoustic",
                    "Free Download",
                    "MP3",
                    "upload_files",
                ],
                "RBA-054": [
                    "https://digitalcollections.nypl.org/",
                    "1700-1799",
                    "Download Options",
                    "upload_files",
                ],
                "RBA-055": ["ESA", "TinyWow", "JPG", "upload_files"],
                "RBA-056": ["arXiv", "TinyWow", "DOC", "upload_files"],
                "RBA-057": [
                    "https://www.metmuseum.org/art/collection/search/437674",
                    "Madonna and Child",
                    "Download Image",
                    "https://tinywow.com/pdf/from-jpg",
                    "Create PDF",
                    "upload_files",
                ],
                "RBA-058": ["NOAA Photo Library", "TinyWow", "PDF", "upload_files"],
                "RBA-059": [
                    "Pride and Prejudice (version 5)",
                    "Internet Archive Page",
                    "DOWNLOAD OPTIONS",
                    "OGG VORBIS",
                    ".ogg",
                    "https://tinywow.com/video/ogg-to-wav",
                    "Upload from PC or Mobile",
                    ".wav",
                    "upload_files",
                ],
                "RBA-099": ["CBP CROSS", "J83798", "tariff classification"],
            }
        )
        for index in range(60, 72):
            full_replacement_routes[f"RBA-{index:03d}"] = [
                "Box Note",
                "All changes saved",
            ]
        for index in range(72, 84):
            full_replacement_routes[f"RBA-{index:03d}"] = [
                "New form",
                "Blank form",
                "Type a question",
                "Publish",
                "Submissions",
            ]
        for index in (*range(72, 80), 81, 83):
            full_replacement_routes[f"RBA-{index:03d}"].append("Short answer")
        full_replacement_routes["RBA-080"].append("Long answer")
        full_replacement_routes["RBA-082"].append("Long answer")
        for index in range(84, 96):
            full_replacement_routes[f"RBA-{index:03d}"] = [
                "Blank form",
                "Publish",
                "Responses",
            ]
        for index in (84, 86, 87, 88, 89, 91, 92, 93, 95):
            full_replacement_routes[f"RBA-{index:03d}"].append("Add question")
        self.assertEqual(
            set(full_replacement_routes),
            full_replacement_ids,
        )
        upload_task_ids = {
            "RBA-011",
            *[f"RBA-{index:03d}" for index in range(40, 46)],
            "RBA-049",
            *[f"RBA-{index:03d}" for index in range(51, 60)],
            "RBA-072",
            "RBA-077",
            "RBA-080",
            "RBA-083",
            "RBA-084",
            "RBA-088",
            "RBA-089",
            "RBA-091",
        }
        marked_task_ids = {
            task.task_id
            for task in explicit.tasks
            if "upload_files" in task.confirmed_task
        }
        self.assertEqual(marked_task_ids, upload_task_ids)

        for canonical_task, explicit_task in zip(
            self.catalog.tasks, explicit.tasks, strict=True
        ):
            canonical_data = canonical_task.to_dict()
            explicit_data = explicit_task.to_dict()
            canonical_prompt = canonical_data.pop("confirmed_task")
            explicit_prompt = explicit_data.pop("confirmed_task")
            self.assertEqual(explicit_data, canonical_data, canonical_task.task_id)
            if canonical_task.task_id in full_replacement_ids:
                self.assertFalse(
                    explicit_prompt.startswith(f"{canonical_prompt} "),
                    canonical_task.task_id,
                )
                for visible_label in full_replacement_routes[canonical_task.task_id]:
                    self.assertIn(
                        visible_label,
                        explicit_prompt,
                        canonical_task.task_id,
                    )
                self.assertNotRegex(
                    explicit_prompt,
                    r"\bURLs?\b",
                    canonical_task.task_id,
                )
                self.assertLessEqual(
                    len(explicit_prompt.split()),
                    230,
                    canonical_task.task_id,
                )
                self.assertNotIn(
                    "browser evaluation",
                    explicit_prompt,
                    canonical_task.task_id,
                )
            else:
                self.assertTrue(
                    explicit_prompt.startswith(f"{canonical_prompt} "),
                    canonical_task.task_id,
                )
                self.assertGreater(
                    len(explicit_prompt.split()),
                    len(canonical_prompt.split()) + 20,
                    canonical_task.task_id,
                )
                added_procedure = explicit_prompt[len(canonical_prompt) :]
                self.assertNotRegex(
                    added_procedure,
                    r"\bURLs?\b",
                    canonical_task.task_id,
                )

        conflict_prompt = explicit.by_id("RBA-050").confirmed_task
        self.assertEqual(conflict_prompt.count("Reload latest version"), 1)
        self.assertEqual(conflict_prompt.count("Save update"), 2)

    def test_all_ato_tasks_start_at_the_simulator_root(self) -> None:
        for catalog_path in (
            REPO_ROOT / "tasks" / "tasks.json",
            REPO_ROOT / "tasks" / "tasks-v2.json",
        ):
            catalog = load_catalog(catalog_path)
            ato_tasks = [
                task for task in catalog.tasks if task.environment.adapter == "ato_simulator"
            ]
            self.assertEqual(len(ato_tasks), 9, catalog_path.name)
            for task in ato_tasks:
                self.assertEqual(
                    task.environment.start_url,
                    "https://onlineservicessimulator.ato.gov.au/",
                    task.task_id,
                )


if __name__ == "__main__":
    unittest.main()
