import { Text, Flex, Box, Container, Button, Checkbox } from "@radix-ui/themes";
import { useState, useEffect } from "react";
import CalendarSlots from "./CalendarSlots";
import { createOnboarding, getSlots } from "../../services/cal";
import heroImage from "../../assets/cards/vlm.webp";
import { useNavigate, Link } from "react-router-dom";
import { useAuth } from "react-oidc-context";
import Loader from "../Loader/Loader";

interface OnboardingData {
  useCase: string;
  monthlyUsage: number;
  fileTypes: string;
  referralSource: string;
  addOns: string[];
}

interface TimeSlot {
  start: string;
}

interface CalendarData {
  [date: string]: TimeSlot[];
}

export default function Onboarding() {
  const [formData, setFormData] = useState<OnboardingData>({
    useCase: "",
    monthlyUsage: 6000,
    fileTypes: "",
    referralSource: "",
    addOns: [],
  });
  const [selectedDate, setSelectedDate] = useState<string>("");
  const [selectedTime, setSelectedTime] = useState<string>("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [calendarData, setCalendarData] = useState<CalendarData>({});
  const [calendarLoading, setCalendarLoading] = useState(true);
  const [calendarError, setCalendarError] = useState<string | null>(null);

  const navigate = useNavigate();
  const auth = useAuth();

  useEffect(() => {
    const fetchSlots = async () => {
      try {
        const response = await getSlots();

        if (response.status === "success" && response.data) {
          setCalendarData(response.data);
        } else {
          throw new Error("API returned error status");
        }
      } catch (err) {
        setCalendarError(
          err instanceof Error ? err.message : "Failed to fetch calendar data"
        );
        console.error("Error fetching calendar slots:", err);
      } finally {
        setCalendarLoading(false);
      }
    };

    fetchSlots();
  }, []);

  const handleInputChange = (
    field: keyof OnboardingData,
    value: string | number
  ) => {
    setFormData((prev) => ({
      ...prev,
      [field]: value,
    }));
  };

  const handleAddOnChange = (addOn: string, checked: boolean) => {
    setFormData((prev) => ({
      ...prev,
      addOns: checked
        ? [...prev.addOns, addOn]
        : prev.addOns.filter((item) => item !== addOn),
    }));
  };

  const handleSlotSelect = (date: string, time: string) => {
    setSelectedDate(date);
    setSelectedTime(time);
  };

  const formatNumber = (num: number) => {
    if (num >= 1000000) {
      return "1M+";
    }
    if (num >= 100000) {
      return `${(num / 1000).toFixed(0)}k`;
    }
    if (num >= 1000) {
      return `${(num / 1000).toFixed(0)}k`;
    }
    return num.toString();
  };

  const getUserTimezone = () => {
    return Intl.DateTimeFormat().resolvedOptions().timeZone;
  };

  const isFormComplete =
    formData.useCase.trim() !== "" &&
    formData.fileTypes.trim() !== "" &&
    formData.referralSource.trim() !== "" &&
    selectedDate !== "" &&
    selectedTime !== "";

  const completeOnboarding = async () => {
    try {
      setIsSubmitting(true);
      const response = await createOnboarding(
        selectedTime,
        getUserTimezone(),
        formData
      );

      if (response.status === "error") {
        const errorMessage =
          response.error?.message ||
          response.error?.details?.message ||
          "Unknown error occurred";
        throw new Error(errorMessage);
      }

      setIsSubmitting(false);
      navigate("/dashboard", { replace: true });
      return response.status;
    } catch (error) {
      setIsSubmitting(false);
      const errorMessage =
        error instanceof Error
          ? error.message
          : "An error occurred while scheduling your call. Please try again.";

      setSubmitError(errorMessage);
      return null;
    }
  };

  if (calendarLoading) {
    return (
      <Flex
        direction="column"
        justify="center"
        align="center"
        className="min-h-screen"
      >
        <Loader />
      </Flex>
    );
  }

  return (
    <div className="min-h-screen relative">
      <div
        className="fixed inset-0 z-0"
        style={{
          backgroundImage: `url(${heroImage})`,
          backgroundSize: "cover",
          backgroundPosition: "center",
          backgroundRepeat: "no-repeat",
        }}
      />
      <div className="fixed inset-0 z-10 bg-gradient-to-t from-black via-black/90 via-50% to-transparent" />

      <Flex
        direction="column"
        justify="start"
        className="min-h-screen relative z-20"
      >
        <Box asChild className="relative z-20">
          <header className="h-fit z-1 py-5 px-8">
            <Flex align="center" justify="between">
              <Link
                to="/"
                className="flex items-center gap-2 hover:opacity-95 transition-opacity"
              >
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  width="30"
                  height="30"
                  viewBox="0 0 30 30"
                  fill="none"
                >
                  <path
                    d="M7.35 12.225C8.03148 12.4978 8.77803 12.5646 9.4971 12.4171C10.2162 12.2695 10.8761 11.9142 11.3952 11.3952C11.9142 10.8761 12.2695 10.2162 12.4171 9.4971C12.5646 8.77803 12.4978 8.03148 12.225 7.35C13.0179 7.13652 13.7188 6.6687 14.2201 6.01836C14.7214 5.36802 14.9954 4.57111 15 3.75C17.225 3.75 19.4001 4.4098 21.2502 5.64597C23.1002 6.88213 24.5422 8.63914 25.3936 10.6948C26.2451 12.7505 26.2679 15.0125 26.0338 17.1948C25.5998 19.3771 24.5283 21.3816 22.955 22.955C21.3816 24.5283 19.3771 25.5998 17.1948 26.0338C15.0125 26.4679 12.7505 26.2451 10.6948 25.3936C8.63914 24.5422 6.88213 23.1002 5.64597 21.2502C4.4098 19.4001 3.75 17.225 3.75 15C4.57111 14.9954 5.36802 14.7214 6.01836 14.2201C6.6687 13.7188 7.13652 13.0179 7.35 12.225Z"
                    stroke="url(#paint0_linear_236_740)"
                    strokeWidth="3"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                  <defs>
                    <linearGradient
                      id="paint0_linear_236_740"
                      x1="15"
                      y1="3.75"
                      x2="15"
                      y2="26.25"
                      gradientUnits="userSpaceOnUse"
                    >
                      <stop stopColor="white" />
                      <stop offset="1" stopColor="#DCE4DD" />
                    </linearGradient>
                  </defs>
                </svg>
                <Text
                  size="6"
                  weight="bold"
                  className="!text-2xl"
                  trim="start"
                  mt="1px"
                >
                  chunkr
                </Text>
              </Link>
              <Button
                onClick={() => auth.signoutRedirect()}
                className="button-active !text-white"
              >
                Logout
              </Button>
            </Flex>
          </header>
        </Box>

        <Container className="text-white w-full flex xl:flex-row flex-col justify-center items-start gap-12 px-4 sm:px-6 lg:px-8 relative z-20 py-8">
          <Flex
            gap="12"
            className="w-full xl:w-1/2 mx-auto lg:flex-row flex-col lg:gap-12 px-0"
          >
            <Box className="flex-1 min-w-0">
              <div className="bg-black/50 backdrop-blur-lg border border-white/10 rounded-xl p-8 shadow-2xl">
                <Flex direction="column" gap="4" className="text-left">
                  <Flex direction="column" gap="2">
                    <Flex align="center" justify="between">
                      <Text className="font-semibold !text-2xl">
                        Onboarding
                      </Text>
                      <Text className="font-medium text-sm">
                        <span className="text-red-500">*</span> Required Fields
                      </Text>
                    </Flex>
                    <Text
                      size="5"
                      weight="medium"
                      className="feature-left-box-subtitle !text-base !text-white/70"
                    >
                      Start your onboarding and get instant access to our
                      platform
                    </Text>
                  </Flex>
                  {submitError && (
                    <div className="bg-red-500/20 border border-red-500 border-l-4 pl-3 py-2 rounded-md">
                      <Text size="2" className="text-red-600">
                        {submitError}
                      </Text>
                    </div>
                  )}
                  <Flex direction="column" gap="6" className="text-left">
                    <Flex direction="column" gap="1">
                      <Text
                        size="3"
                        weight="medium"
                        className="feature-box-description text-white/90 py-2"
                      >
                        What's your primary use case?{" "}
                        <span className="text-red-500">*</span>
                      </Text>
                      <textarea
                        placeholder="Describe your use case. For eg: RAG applications, document workflows, structured extraction, ..."
                        value={formData.useCase}
                        onChange={(e) =>
                          handleInputChange("useCase", e.target.value)
                        }
                        className="w-full p-4 text-base border border-white/20 rounded-lg bg-white/5 text-white transition-all duration-200 outline-none box-border resize-none min-h-12 font-inherit leading-relaxed placeholder:text-white/50 focus:border-gray-400 focus:bg-white/8 focus:shadow-[0_0_0_3px_rgba(139,139,139,0.1)] hover:border-white/30 hover:bg-white/7"
                        rows={2}
                      />
                    </Flex>

                    <Flex direction="column" gap="1">
                      <Text
                        size="3"
                        weight="medium"
                        className="feature-box-description text-white/90 py-2"
                      >
                        Expected monthly usage:{" "}
                        {formatNumber(formData.monthlyUsage)} pages{" "}
                        <span className="text-red-500">*</span>
                      </Text>
                      <div className="space-y-3">
                        <input
                          type="range"
                          min="1000"
                          max="1000000"
                          step="1000"
                          value={formData.monthlyUsage}
                          onChange={(e) =>
                            handleInputChange(
                              "monthlyUsage",
                              parseInt(e.target.value)
                            )
                          }
                          className="w-full h-2 rounded-lg appearance-none cursor-pointer transition-all duration-200
                        [&::-webkit-slider-thumb]:appearance-none 
                        [&::-webkit-slider-thumb]:h-5 
                        [&::-webkit-slider-thumb]:w-5 
                        [&::-webkit-slider-thumb]:rounded-full 
                        [&::-webkit-slider-thumb]:bg-white 
                        [&::-webkit-slider-thumb]:cursor-pointer 
                        [&::-webkit-slider-thumb]:shadow-lg 
                        [&::-webkit-slider-thumb]:border-none
                        [&::-webkit-slider-thumb]:transition-transform
                        [&::-webkit-slider-thumb:hover]:scale-110
                        [&::-moz-range-thumb]:h-5 
                        [&::-moz-range-thumb]:w-5 
                        [&::-moz-range-thumb]:rounded-full 
                        [&::-moz-range-thumb]:bg-white 
                        [&::-moz-range-thumb]:cursor-pointer 
                        [&::-moz-range-thumb]:shadow-lg 
                        [&::-moz-range-thumb]:border-none"
                          style={{
                            background: `linear-gradient(to right, white 0%, white ${((formData.monthlyUsage - 1000) / 1000000) * 100
                              }%, rgba(255,255,255,0.2) ${((formData.monthlyUsage - 1000) / 1000000) * 100
                              }%, rgba(255,255,255,0.2) 100%)`,
                          }}
                        />
                        <div className="flex justify-between text-sm text-white/60">
                          <span>1k</span>
                          <span>250k</span>
                          <span>500k</span>
                          <span>750k</span>
                          <span>1M+</span>
                        </div>
                      </div>
                    </Flex>

                    <Flex direction="column" gap="1">
                      <Text
                        size="3"
                        weight="medium"
                        className="feature-box-description text-white/90 py-2"
                      >
                        What file types will you be processing?{" "}
                        <span className="text-red-500">*</span>
                      </Text>
                      <input
                        type="text"
                        placeholder="PDF, Excel, Word Docs, PowerPoint, Images, ..."
                        value={formData.fileTypes}
                        onChange={(e) =>
                          handleInputChange("fileTypes", e.target.value)
                        }
                        className="w-full py-2 px-3 text-base border border-white/20 rounded-lg bg-white/5 text-white transition-all duration-200 outline-none box-border placeholder:text-white/50 focus:border-gray-400 focus:bg-white/8 focus:shadow-[0_0_0_3px_rgba(139,139,139,0.1)] hover:border-white/30 hover:bg-white/7"
                      />
                    </Flex>

                    <Flex direction="column" gap="1">
                      <Text
                        size="3"
                        weight="medium"
                        className="feature-box-description text-white/90 py-2"
                      >
                        How did you hear about us?{" "}
                        <span className="text-red-500">*</span>
                      </Text>
                      <select
                        value={formData.referralSource}
                        onChange={(e) =>
                          handleInputChange("referralSource", e.target.value)
                        }
                        className="w-full py-2 px-3 text-base border border-white/20 rounded-lg bg-white/5 text-white transition-all duration-200 outline-none box-border focus:border-gray-400 focus:bg-white/8 focus:shadow-[0_0_0_3px_rgba(139,139,139,0.1)] hover:border-white/30 hover:bg-white/7"
                      >
                        <option value="" className="bg-gray-800 text-white">
                          Select an option
                        </option>
                        <option
                          value="search_engine"
                          className="bg-gray-800 text-white"
                        >
                          Search Engine (Google, Bing, etc.)
                        </option>
                        <option
                          value="social_media"
                          className="bg-gray-800 text-white"
                        >
                          Social Media
                        </option>
                        <option
                          value="word_of_mouth"
                          className="bg-gray-800 text-white"
                        >
                          Word of Mouth
                        </option>
                        <option
                          value="blog_article"
                          className="bg-gray-800 text-white"
                        >
                          Blog/Article
                        </option>
                        <option
                          value="conference_event"
                          className="bg-gray-800 text-white"
                        >
                          Conference/Event
                        </option>
                        <option
                          value="github"
                          className="bg-gray-800 text-white"
                        >
                          GitHub
                        </option>
                        <option
                          value="linkedin"
                          className="bg-gray-800 text-white"
                        >
                          LinkedIn
                        </option>
                        <option
                          value="twitter"
                          className="bg-gray-800 text-white"
                        >
                          Twitter/X
                        </option>
                        <option
                          value="reddit"
                          className="bg-gray-800 text-white"
                        >
                          Reddit
                        </option>
                        <option
                          value="youtube"
                          className="bg-gray-800 text-white"
                        >
                          YouTube
                        </option>
                        <option
                          value="podcast"
                          className="bg-gray-800 text-white"
                        >
                          Podcast
                        </option>
                        <option
                          value="other"
                          className="bg-gray-800 text-white"
                        >
                          Other
                        </option>
                      </select>
                    </Flex>

                    <Flex direction="column" gap="1">
                      <Text
                        size="3"
                        weight="medium"
                        className="feature-box-description text-white/90 py-2"
                      >
                        Are you interested in any of these add-ons?
                      </Text>
                      <div className="grid grid-cols-2 gap-3">
                        {[
                          { value: "on_prem", label: "On-Prem/Self-Host" },
                          { value: "sso_saml", label: "SSO/SAML" },
                          { value: "zdr_baa", label: "ZDR/BAA Agreements" },
                          { value: "custom_slas", label: "Custom SLAs" },
                        ].map((addOn) => (
                          <div
                            key={addOn.value}
                            className="flex items-center space-x-2"
                          >
                            <Checkbox
                              id={addOn.value}
                              checked={formData.addOns.includes(addOn.value)}
                              onCheckedChange={(checked) =>
                                handleAddOnChange(addOn.value, checked === true)
                              }
                              color="gray"
                              className="rounded-sm focus:ring-0 focus:outline-none"
                            />
                            <label
                              htmlFor={addOn.value}
                              className="cursor-pointer"
                            >
                              <Text as="span" size="2">
                                {addOn.label}
                              </Text>
                            </label>
                          </div>
                        ))}
                      </div>
                    </Flex>

                    <Flex direction="column" gap="1">
                      <Text
                        size="3"
                        weight="medium"
                        className="feature-box-description text-white/90 py-2"
                      >
                        Onboarding Call <span className="text-red-500">*</span>
                      </Text>
                      <div
                        className={`w-full mx-auto overflow-auto rounded-md transition-all duration-300 ${selectedDate && selectedTime
                          ? "h-auto max-h-none"
                          : "h-[240px] max-h-[240px]"
                          }`}
                      >
                        <CalendarSlots
                          calendarData={calendarData}
                          calendarError={calendarError}
                          onSlotSelect={handleSlotSelect}
                          selectedDate={selectedDate}
                          selectedTime={selectedTime}
                          showScrollable={true}
                        />
                      </div>
                    </Flex>

                    <Box className="pt-2">
                      <Button
                        onClick={completeOnboarding}
                        disabled={!isFormComplete || isSubmitting}
                        className="button-active !w-full !h-10 !text-white disabled:opacity-50 disabled:cursor-not-allowed"
                      >
                        {isSubmitting
                          ? "Routing to Dashboard"
                          : "Schedule Call"}
                      </Button>
                    </Box>
                  </Flex>
                </Flex>
              </div>
            </Box>
          </Flex>
        </Container>
      </Flex>
    </div>
  );
}
